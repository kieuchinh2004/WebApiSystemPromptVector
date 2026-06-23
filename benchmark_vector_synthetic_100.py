"""Benchmark the WebAPI_Vector pipeline on the APC v1.3 synthetic 100 CSV."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parent
DATASET = Path(r"R:\DoAn\AI\Dataset\Test_folder\apc_v13_synthetic_100_20260623.csv")
PROMPT = BASE / "prompts" / "system_prompt_vector_v1.md"
ROOT_OUT = BASE / "accuracy_outputs" / "vector_synthetic_100"
API_PORT = 8010
LLAMA_PORT = 28285
API_URL = f"http://127.0.0.1:{API_PORT}"

PROFILES = [
    "qwen2_5_1_5b.py",
    "qwen2_5_3b.py",
    "qwen2_5_7b.py",
    "lfm2_5_1_2b.py",
    "qwen3_1_7b.py",
    "smollm2_1_7b.py",
]

VALID_LABELS = tuple(f"L{i}" for i in range(7)) + ("Thieu context",)


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    upper = text.upper()
    if upper in {f"L{i}" for i in range(7)}:
        return upper
    folded = unicodedata.normalize("NFD", text)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    folded = " ".join(folded.lower().split())
    if folded in {"thieu context", "needs context", "need context"}:
        return "Thieu context"
    return None


def http_json(path: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API_URL + path,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(process: subprocess.Popen, seconds: int = 240) -> tuple[bool, str, dict[str, Any]]:
    deadline = time.time() + seconds
    last = "not reachable"
    while time.time() < deadline:
        if process.poll() is not None:
            return False, f"API exited with code {process.returncode}", {}
        try:
            health = http_json("/health", timeout=5)
            if health.get("llama_server_status") == "ok":
                return True, "ready", http_json("/info", timeout=5)
            last = str(health)
        except Exception as exc:
            last = str(exc)
        time.sleep(2)
    return False, f"timeout; last={last}", {}


def stop_api(process: subprocess.Popen) -> None:
    try:
        http_json("/server/action", {"action": "stop"}, timeout=30)
    except Exception:
        pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    time.sleep(2)


def load_samples() -> list[dict[str, Any]]:
    with DATASET.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            label = normalize_label(row.get("label"))
            text = (row.get("text") or "").strip()
            if text and label in VALID_LABELS:
                rows.append({
                    "id": row.get("id") or str(len(rows) + 1),
                    "text": text,
                    "true_label": label,
                })
    if not rows:
        raise ValueError(f"No valid rows in {DATASET}")
    return rows


def classify_one(sample: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        data = http_json("/classify", {"prompt": sample["text"]}, timeout=timeout)
        latency = time.perf_counter() - started
        pred = normalize_label(data.get("level"))
        status = "OK" if pred in VALID_LABELS else "INVALID_PREDICTION"
        return {
            **sample,
            "predicted_label": pred or "",
            "raw_predicted_label": data.get("level", ""),
            "correct": pred == sample["true_label"],
            "status": status,
            "confidence": data.get("confidence"),
            "margin": data.get("margin"),
            "accept": data.get("accept"),
            "vector": data.get("vector"),
            "explanation": data.get("explanation"),
            "latency_seconds": round(latency, 3),
        }
    except Exception as exc:
        return {
            **sample,
            "predicted_label": "",
            "raw_predicted_label": "",
            "correct": False,
            "status": "REQUEST_ERROR",
            "confidence": "",
            "margin": "",
            "accept": "",
            "vector": "",
            "explanation": repr(exc),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    valid = [r for r in results if r["status"] == "OK"]
    correct = sum(1 for r in valid if r["correct"])
    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for label in VALID_LABELS:
        support = sum(1 for r in results if r["true_label"] == label)
        pred_count = sum(1 for r in valid if r["predicted_label"] == label)
        tp = sum(1 for r in valid if r["true_label"] == label and r["predicted_label"] == label)
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1_values.append(f1)
        per_class[label] = {
            "support": support,
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "total_dataset_rows": total,
        "valid_predictions": len(valid),
        "failed_or_invalid_predictions": total - len(valid),
        "correct_predictions": correct,
        "classification_accuracy": correct / len(valid) if valid else 0.0,
        "end_to_end_accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
    }


def run_profile(profile: str) -> dict[str, Any]:
    out = ROOT_OUT / Path(profile).stem
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "API_PORT": str(API_PORT),
        "LLAMA_PORT": str(LLAMA_PORT),
        "MODEL_CONFIG_FILE": profile,
        "PROMPT_FILE": str(PROMPT),
        "MAX_TOKENS": env.get("MAX_TOKENS", "192"),
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
    })
    print(f"[{datetime.now():%H:%M:%S}] START {profile}", flush=True)
    api_log = out / "api.log"
    result: dict[str, Any] = {"profile": profile, "status": "startup_error"}
    with api_log.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(["python", "main.py"], cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            ready, message, info = wait_ready(process)
            result.update({"startup_message": message, "api_info": info})
            if not ready:
                return result
            samples = load_samples()
            workers = max(1, min(4, int(info.get("parallel_slots", 1))))
            details: list[dict[str, Any]] = []
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(classify_one, sample) for sample in samples]
                for idx, future in enumerate(as_completed(futures), start=1):
                    details.append(future.result())
                    if idx % 25 == 0:
                        errors = sum(1 for r in details if r["status"] != "OK")
                        correct = sum(1 for r in details if r["correct"])
                        print(f"Progress {profile}: {idx}/{len(samples)} correct={correct} errors={errors}", flush=True)
            details.sort(key=lambda row: row["id"])
            metrics = summarize(details)
            detail_path = out / f"accuracy_details_{datetime.now():%Y%m%d_%H%M%S}.csv"
            with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(details[0].keys()))
                writer.writeheader()
                writer.writerows(details)
            result.update({
                "status": "ok" if metrics["failed_or_invalid_predictions"] == 0 else "completed_with_errors",
                "metrics": metrics,
                "detail_file": str(detail_path),
                "benchmark_elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            return result
        finally:
            stop_api(process)
            print(f"[{datetime.now():%H:%M:%S}] END {profile}: {result.get('status')}", flush=True)


def save_report(results: list[dict[str, Any]]) -> None:
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    json_path = ROOT_OUT / "benchmark_vector_summary.json"
    csv_path = ROOT_OUT / "benchmark_vector_summary.csv"
    md_path = ROOT_OUT / "benchmark_vector_report.md"
    json_path.write_text(json.dumps({
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": str(DATASET),
        "prompt_file": str(PROMPT),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for item in results:
        metrics = item.get("metrics", {})
        info = item.get("api_info", {})
        rows.append({
            "profile": item["profile"],
            "model": info.get("model", ""),
            "status": item.get("status", ""),
            "classification_accuracy": metrics.get("classification_accuracy", ""),
            "end_to_end_accuracy": metrics.get("end_to_end_accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "total_samples": metrics.get("total_dataset_rows", ""),
            "valid_predictions": metrics.get("valid_predictions", ""),
            "failed_or_invalid_predictions": metrics.get("failed_or_invalid_predictions", ""),
            "elapsed_seconds": item.get("benchmark_elapsed_seconds", ""),
            "error": item.get("startup_message", ""),
        })
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# WebAPI_Vector benchmark - APC synthetic 100",
        "",
        f"- Dataset: `{DATASET}`",
        f"- Prompt: `{PROMPT}`",
        "",
        "| Profile | Model | Status | Accuracy(valid) | End-to-end | Macro F1 | Valid/Total |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(v: Any) -> str:
            return f"{float(v):.4f}" if isinstance(v, (int, float)) or str(v).replace(".", "", 1).isdigit() else "-"
        lines.append(
            f"| {row['profile']} | {row['model'] or '-'} | {row['status']} | "
            f"{fmt(row['classification_accuracy'])} | {fmt(row['end_to_end_accuracy'])} | "
            f"{fmt(row['macro_f1'])} | {row['valid_predictions']}/{row['total_samples']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reports:\n{csv_path}\n{json_path}\n{md_path}", flush=True)


def main() -> int:
    for path in (DATASET, PROMPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    results = []
    for profile in PROFILES:
        results.append(run_profile(profile))
        save_report(results)
    return 0 if all(item.get("status") == "ok" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
