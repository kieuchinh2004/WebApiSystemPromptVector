"""Benchmark APC vector extraction feature-by-feature.

This script is intentionally separate from the normal level-accuracy benchmark.
It calls /classify_debug and records:

    - llm_p1/llm_p2/llm_a1...llm_r10
    - final_p1/final_p2/final_a1...final_r10

If the input CSV contains true vector columns, it also calculates per-feature
accuracy.  Accepted true-column names:

    p1,p2,a1..a7,d1..d7,r1..r10
    true_p1,true_p2,true_a1..true_r10
    gold_p1,gold_p2,gold_a1..gold_r10

The input prompt column can be text, prompt, or student_prompt.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from vector_schema import VECTOR_FIELDS


BASE = Path(__file__).resolve().parent
DATASET = Path(os.environ.get("DATASET", r"R:\DoAn\AI\Dataset\Test_folder\apc_v13_synthetic_100_20260623.csv"))
PROMPT = Path(os.environ.get("PROMPT_FILE", str(BASE / "prompts" / "system_prompt_vector_v1.md")))
ROOT_OUT = BASE / "accuracy_outputs" / "vector_feature_check"
API_PORT = int(os.environ.get("API_PORT", "8011"))
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "28286"))
API_URL = f"http://127.0.0.1:{API_PORT}"

PROFILES = [
    "qwen2_5_1_5b.py",
    "qwen2_5_3b.py",
    "qwen2_5_7b.py",
    "lfm2_5_1_2b.py",
    "qwen3_1_7b.py",
    "smollm2_1_7b.py",
]

FEATURE_KEYS = [key for key, _, _ in VECTOR_FIELDS]


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


def parse_truth(row: dict[str, str]) -> dict[str, int | None]:
    truth: dict[str, int | None] = {}
    for key in FEATURE_KEYS:
        raw = None
        for column in (key, f"true_{key}", f"gold_{key}"):
            if column in row and str(row[column]).strip() != "":
                raw = str(row[column]).strip()
                break
        if raw is None:
            truth[key] = None
            continue
        try:
            value = float(raw)
        except ValueError:
            truth[key] = None
            continue
        truth[key] = 1 if value >= 0.5 else 0
    return truth


def load_samples() -> list[dict[str, Any]]:
    with DATASET.open("r", encoding="utf-8-sig", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(csv.DictReader(handle), start=1):
            text = (row.get("text") or row.get("prompt") or row.get("student_prompt") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "id": row.get("id") or str(index),
                    "text": text,
                    "label": row.get("label", ""),
                    "truth": parse_truth(row),
                }
            )
    if not rows:
        raise ValueError(f"No valid prompt rows in {DATASET}")
    return rows


def detail_to_flat(items: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in items or []:
        key = item.get("key")
        if key:
            output[f"{prefix}_{key}"] = item.get("value", "")
    return output


def classify_one(sample: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    started = time.perf_counter()
    base = {
        "id": sample["id"],
        "text": sample["text"],
        "label": sample.get("label", ""),
    }
    for key in FEATURE_KEYS:
        value = sample["truth"].get(key)
        base[f"true_{key}"] = "" if value is None else value

    try:
        data = http_json("/classify_debug", {"prompt": sample["text"]}, timeout=timeout)
        row = {
            **base,
            "status": "OK",
            "level": data.get("level", ""),
            "confidence": data.get("confidence", ""),
            "accept": data.get("accept", ""),
            "vector": data.get("vector", ""),
            "explanation": data.get("explanation", ""),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }
        row.update(detail_to_flat(data.get("llm_vector_detail", []), "llm"))
        row.update(detail_to_flat(data.get("final_vector_detail", []), "final"))
        for key in FEATURE_KEYS:
            truth = sample["truth"].get(key)
            pred_raw = row.get(f"final_{key}", "")
            pred = 1 if isinstance(pred_raw, (int, float)) and float(pred_raw) >= 0.5 else 0
            row[f"correct_{key}"] = "" if truth is None else int(pred == truth)
        return row
    except Exception as exc:
        row = {
            **base,
            "status": "REQUEST_ERROR",
            "level": "",
            "confidence": "",
            "accept": "",
            "vector": "",
            "explanation": repr(exc),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }
        for prefix in ("llm", "final"):
            for key in FEATURE_KEYS:
                row[f"{prefix}_{key}"] = ""
        for key in FEATURE_KEYS:
            row[f"correct_{key}"] = ""
        return row


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in details if row["status"] == "OK"]
    per_feature: dict[str, Any] = {}
    feature_acc_values: list[float] = []
    for key in FEATURE_KEYS:
        scored = [row for row in valid if row.get(f"correct_{key}") != ""]
        correct = sum(int(row[f"correct_{key}"]) for row in scored)
        accuracy = correct / len(scored) if scored else None
        if accuracy is not None:
            feature_acc_values.append(accuracy)
        per_feature[key] = {
            "scored": len(scored),
            "correct": correct,
            "accuracy": accuracy,
        }
    return {
        "total_rows": len(details),
        "valid_rows": len(valid),
        "error_rows": len(details) - len(valid),
        "macro_vector_accuracy": sum(feature_acc_values) / len(feature_acc_values) if feature_acc_values else None,
        "per_feature": per_feature,
    }


def run_profile(profile: str) -> dict[str, Any]:
    out = ROOT_OUT / Path(profile).stem
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "API_PORT": str(API_PORT),
            "LLAMA_PORT": str(LLAMA_PORT),
            "MODEL_CONFIG_FILE": profile,
            "PROMPT_FILE": str(PROMPT),
            "MAX_TOKENS": env.get("MAX_TOKENS", "192"),
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
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
                        errors = sum(1 for row in details if row["status"] != "OK")
                        print(f"Progress {profile}: {idx}/{len(samples)} errors={errors}", flush=True)
            details.sort(key=lambda row: str(row["id"]))
            metrics = summarize(details)
            detail_path = out / f"vector_feature_details_{datetime.now():%Y%m%d_%H%M%S}.csv"
            fieldnames = list(details[0].keys())
            with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(details)
            result.update(
                {
                    "status": "ok" if metrics["error_rows"] == 0 else "completed_with_errors",
                    "metrics": metrics,
                    "detail_file": str(detail_path),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            return result
        finally:
            stop_api(process)
            print(f"[{datetime.now():%H:%M:%S}] END {profile}: {result.get('status')}", flush=True)


def save_report(results: list[dict[str, Any]]) -> None:
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    json_path = ROOT_OUT / "benchmark_vector_feature_summary.json"
    csv_path = ROOT_OUT / "benchmark_vector_feature_summary.csv"
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": str(DATASET),
        "prompt_file": str(PROMPT),
        "feature_keys": FEATURE_KEYS,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for item in results:
        metrics = item.get("metrics", {})
        info = item.get("api_info", {})
        rows.append(
            {
                "profile": item["profile"],
                "model": info.get("model", ""),
                "status": item.get("status", ""),
                "total_rows": metrics.get("total_rows", ""),
                "valid_rows": metrics.get("valid_rows", ""),
                "error_rows": metrics.get("error_rows", ""),
                "macro_vector_accuracy": metrics.get("macro_vector_accuracy", ""),
                "detail_file": item.get("detail_file", ""),
                "elapsed_seconds": item.get("elapsed_seconds", ""),
                "startup_message": item.get("startup_message", ""),
            }
        )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reports:\n{csv_path}\n{json_path}", flush=True)


def main() -> int:
    for path in (DATASET, PROMPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    selected = [p.strip() for p in os.environ.get("PROFILES", "").split(",") if p.strip()] or PROFILES
    results = []
    for profile in selected:
        results.append(run_profile(profile))
        save_report(results)
    return 0 if all(item.get("status") == "ok" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
