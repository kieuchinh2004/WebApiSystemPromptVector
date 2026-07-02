"""LLM vector extraction + deterministic APC v4 scoring."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

import config as cfg
from apc_v4_engine import PROMPT_VECTOR_SIZE, VECTOR_SIZE, LEGACY_PROMPT_VECTOR_SIZE, LEGACY_TRANSACTION_VECTOR_SIZE, compute_apc_v4
from rule_based_extractor import extract_rule_vector, merge_rule_and_llm_vectors, rule_evidence_as_dict


_system_prompt_cache: str | None = None


class VectorParseError(ValueError):
    """Raised when the LLM output cannot be trusted as a valid APC vector."""


def get_system_prompt() -> str:
    """Read and cache the system prompt."""
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache

    prompt_file = cfg.PROMPT_FILE
    if not prompt_file:
        raise FileNotFoundError("Không tìm thấy system prompt. Thiết lập PROMPT_FILE trong .env")

    with open(prompt_file, "r", encoding="utf-8-sig") as f:
        _system_prompt_cache = f.read().strip()
    return _system_prompt_cache


def _clean_json_markdown(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
    elif content.startswith("```"):
        content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
    return content.strip()


def _coerce_feature_list(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise VectorParseError(f"'{field_name}' must be a list")
    if len(value) not in (PROMPT_VECTOR_SIZE, VECTOR_SIZE, LEGACY_PROMPT_VECTOR_SIZE, LEGACY_TRANSACTION_VECTOR_SIZE):
        raise VectorParseError(
            f"'{field_name}' must contain exactly one of {PROMPT_VECTOR_SIZE}, {LEGACY_PROMPT_VECTOR_SIZE}, {LEGACY_TRANSACTION_VECTOR_SIZE}, or {VECTOR_SIZE} values, got {len(value)}"
        )

    features: list[float] = []
    for idx, item in enumerate(value):
        if isinstance(item, bool):
            number = 1.0 if item else 0.0
        elif isinstance(item, (int, float)):
            number = float(item)
        else:
            raise VectorParseError(f"'{field_name}[{idx}]' must be numeric in [0, 1], got {item!r}")
        if not 0.0 <= number <= 1.0:
            raise VectorParseError(f"'{field_name}[{idx}]' must be in [0, 1], got {number}")
        features.append(number)
    return features


def _coerce_binary_vector(value: Any) -> list[float]:
    if not isinstance(value, str):
        raise VectorParseError("'vector' must be a string")
    vector = "".join(value.split())
    if len(vector) not in (PROMPT_VECTOR_SIZE, VECTOR_SIZE, LEGACY_PROMPT_VECTOR_SIZE, LEGACY_TRANSACTION_VECTOR_SIZE) or not all(c in "01" for c in vector):
        raise VectorParseError(
            f"'vector' must contain exactly one of {PROMPT_VECTOR_SIZE}, {LEGACY_PROMPT_VECTOR_SIZE}, {LEGACY_TRANSACTION_VECTOR_SIZE}, or {VECTOR_SIZE} binary characters, "
            f"got {len(vector)}: {vector!r}"
        )
    return [float(c) for c in vector]


def parse_vector_from_content(content: str) -> tuple[list[float], str, dict[str, Any]]:
    """Parse a strict JSON vector response from the LLM.

    Accepted payloads:
      {"features": [60 numeric values in [0,1]], "explanation": "..."}
      {"vector_values": [60 numeric values in [0,1]], "explanation": "..."}
      {"vector": "60 binary chars", "explanation": "..."}

    Legacy 26D prompt-only and 44D transaction vectors are accepted and mapped by the engine.

    No pad/truncate fallback is used; wrong-length output is unsafe because it
    shifts feature meanings.
    """

    cleaned = _clean_json_markdown(content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise VectorParseError(f"LLM output is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise VectorParseError("LLM JSON output must be an object")

    explanation = data.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = "Vector extracted without explanation."
    explanation = explanation.strip()

    if "features" in data:
        return _coerce_feature_list(data["features"], "features"), explanation, data
    if "vector_values" in data:
        return _coerce_feature_list(data["vector_values"], "vector_values"), explanation, data
    if "vector" in data:
        return _coerce_binary_vector(data["vector"]), explanation, data

    raise VectorParseError("LLM JSON must include one of: 'features', 'vector_values', or 'vector'")


def _build_messages(
    system_prompt: str,
    student_prompt: str,
    ai_output: str | None = None,
    repair_error: str | None = None,
) -> list[dict[str, str]]:
    if ai_output and ai_output.strip():
        user_content = f"STUDENT_PROMPT:\n{student_prompt}\n\nAI_OUTPUT:\n{ai_output.strip()}"
    else:
        user_content = f"student_prompt: {student_prompt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if repair_error:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous output was invalid: "
                    f"{repair_error}. Return ONLY one valid JSON object with either "
                    f"'features' as exactly one of {PROMPT_VECTOR_SIZE}, {LEGACY_PROMPT_VECTOR_SIZE}, {LEGACY_TRANSACTION_VECTOR_SIZE}, or {VECTOR_SIZE} numbers in [0,1], or 'vector' as exactly "
                    f"one of {PROMPT_VECTOR_SIZE}, {LEGACY_PROMPT_VECTOR_SIZE}, {LEGACY_TRANSACTION_VECTOR_SIZE}, or {VECTOR_SIZE} binary characters. No markdown, no prose."
                ),
            }
        )
    return messages


def _call_llama(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": cfg.MODEL_NAME,
        "messages": messages,
        "temperature": cfg.TEMPERATURE,
        "top_p": cfg.TOP_P,
        "max_tokens": cfg.MAX_TOKENS,
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        cfg.LLAMA_CHAT_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    with urllib.request.urlopen(req, timeout=120.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"], body


def classify_with_retry(student_prompt: str, ai_output: str | None = None) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Extract vector with the LLM, then run deterministic APC v4 scoring."""

    system_prompt = get_system_prompt()
    t0 = time.time()
    default_metrics = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
        "tokens_per_second": 0.0,
    }

    last_exception: Exception | None = None
    last_parse_error: str | None = None
    raw_content = ""

    for attempt in range(cfg.RETRY_COUNT):
        try:
            messages = _build_messages(system_prompt, student_prompt, ai_output, last_parse_error)
            raw_content, body = _call_llama(messages)
            print(f"[classifier] Raw LLM content attempt={attempt + 1}: {repr(raw_content)}")

            llm_vector_values, explanation, parsed_payload = parse_vector_from_content(raw_content)
            rule_evidence = extract_rule_vector(student_prompt, ai_output)
            final_vector_values = merge_rule_and_llm_vectors(llm_vector_values, rule_evidence)
            engine_res = compute_apc_v4(final_vector_values)
            engine_res["explanation"] = (
                "Rubric-grounded vector result. "
                f"LLM extraction: {explanation}. "
                "Rule-based evidence was merged as deterministic lower-bound overrides for observable features."
            )
            engine_res["llm_payload"] = parsed_payload
            engine_res["llm_vector_values"] = [round(v, 4) for v in llm_vector_values]
            engine_res["final_vector_values"] = [round(v, 4) for v in final_vector_values]
            engine_res["rule_evidence"] = rule_evidence_as_dict(rule_evidence)

            latency = time.time() - t0
            usage = body.get("usage", {})
            timings = body.get("timings", {})
            metrics = {
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "tokens_per_second": timings.get("predicted_per_second", 0.0),
            }
            return engine_res, latency, metrics

        except VectorParseError as exc:
            last_exception = exc
            last_parse_error = str(exc)
            print(f"[classifier][WARNING] Parse attempt {attempt + 1}/{cfg.RETRY_COUNT} failed: {exc}")
            time.sleep(0.25)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_exception = exc
            print(f"[classifier][WARNING] Request attempt {attempt + 1}/{cfg.RETRY_COUNT} failed: {exc}")
            time.sleep(0.5)

    latency = time.time() - t0
    return {
        "level": "Loi phan tich",
        "candidate_level": "N/A",
        "predicted_level_math": "N/A",
        "score": 0.0,
        "confidence": 0.0,
        "margin": 0.0,
        "accept": 0,
        "scores": [0.0] * 6,
        "gatings": [False] * 6,
        "probs": [0.0] * 6,
        "all_probs": [0.0] * 6,
        "is_coding": False,
        "has_context": False,
        "vector_str": "0" * VECTOR_SIZE,
        "vector_values": [0.0] * VECTOR_SIZE,
        "final_vector_values": [0.0] * VECTOR_SIZE,
        "llm_vector_values": [0.0] * VECTOR_SIZE,
        "rule_evidence": {},
        "constraint_warnings": [],
        "transaction_status": "extractor_error",
        "final_status": "extractor_error",
        "requires_student_confirmation": False,
        "confirmation_reasons": [],
        "alignment_score": None,
        "mismatch_score": None,
        "output_diagnostics": {},
        "explanation": f"Lỗi sau {cfg.RETRY_COUNT} lượt thử: {last_exception}. Raw={raw_content[:200]!r}",
    }, latency, default_metrics
