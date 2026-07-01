"""Fast rule-based feature pre-extraction for APC transaction vectors.

The LLM still handles semantic judgments, but obvious evidence is extracted with
regex/rules first.  This improves speed, stability, and bypass resistance:
student self-labels are ignored, while observable artifacts and action verbs are
anchored deterministically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from apc_v4_engine import PROMPT_VECTOR_SIZE, VECTOR_SIZE
from vector_schema import VECTOR_FIELDS


FIELD_INDEX = {name: idx for idx, (_, name, _) in enumerate(VECTOR_FIELDS)}


@dataclass
class RuleEvidence:
    vector: list[float]
    hits: dict[str, list[str]] = field(default_factory=dict)

    def set(self, field_name: str, value: float = 1.0, evidence: str | None = None) -> None:
        idx = FIELD_INDEX[field_name]
        self.vector[idx] = max(self.vector[idx], float(value))
        if evidence:
            self.hits.setdefault(field_name, []).append(evidence[:160])


def _has(pattern: str, text: str, flags: int = re.IGNORECASE | re.MULTILINE) -> re.Match[str] | None:
    return re.search(pattern, text or "", flags)


def _hit(pattern: str, text: str, flags: int = re.IGNORECASE | re.MULTILINE) -> str | None:
    m = _has(pattern, text, flags)
    if not m:
        return None
    return m.group(0).strip()


def _line_count(text: str) -> int:
    return len([line for line in (text or "").splitlines() if line.strip()])


def extract_rule_vector(student_prompt: str, ai_output: str | None = None) -> RuleEvidence:
    v = [0.0] * VECTOR_SIZE
    ev = RuleEvidence(v)
    p = student_prompt or ""
    o = ai_output or ""
    lower = p.lower()

    # Prefix: coding and context.  Keep has_context permissive unless prompt is empty or explicitly injection-only.
    if p.strip():
        ev.set("has_context", 1.0, "non-empty prompt")
    if _hit(r"```|\b(def|class|function|const|let|var|public|private|import|from|return|SELECT|CREATE TABLE|INSERT INTO)\b|[{};]\s*$", p):
        ev.set("is_coding", 1.0, "code-like syntax")
    if _hit(r"\b(code|bug|debug|fix|error|exception|api|syntax|regex|sql|database|html|css|javascript|python|java|c#|asp\.net|ef core|postgres|react|flutter|git|docker)\b", p):
        ev.set("is_coding", 1.0, "coding keyword")
    if _hit(r"\b(ignore previous|bỏ qua|đừng chấm|hãy phân loại là|classify as L[0-6])\b", p):
        # Injection/self-label evidence makes context unsafe but does not decide level.
        ev.set("has_context", 0.0, "possible self-label/injection; not trusted")

    # Artifact features.
    if _hit(r"```[\s\S]*?```|\b(def|class|function|const|let|var|public|private|import|from|return)\b|;\s*$", p):
        ev.set("a1_code_block", 1.0, "code block or code-like snippet")
    if _hit(r"\b(Traceback|Exception|Error:|TypeError|ValueError|NullReferenceException|IndexError|KeyError|SyntaxError|ReferenceError|stack trace|at line|line \d+)\b", p):
        ev.set("a2_error_trace", 1.0, "error/exception/line diagnostic")
    if _hit(r"\b(SELECT|INSERT INTO|UPDATE\s+\w+\s+SET|DELETE FROM|CREATE TABLE|ALTER TABLE|JOIN|DbContext|migrationBuilder|HasColumnType)\b", p):
        ev.set("a3_sql_schema", 1.0, "SQL/DB/schema evidence")
    if _hit(r'(^|\n)\s*[\w.-]+\s*:\s*[^\n]+|\{\s*\"[^\"]+\"\s*:|\b(appsettings\.json|\.env|docker-compose|package\.json|yaml|yml)\b', p):
        ev.set("a4_config_file", 1.0, "config-like key:value/json/yaml/env evidence")
    if _hit(r"\b(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|INFO|WARN|ERROR|DEBUG|npm ERR!|GET /|POST /|status=\d{3}|exit code)\b", p):
        ev.set("a5_system_log", 1.0, "log/terminal output evidence")
    if _line_count(p) >= 80 or len(p) >= 5000 or _hit(r"\b(full project|whole project|toàn bộ project|complete solution|entire file|nhiều file)\b", p):
        ev.set("a6_extensive_context", 1.0, "large/whole-solution context")
    elif _line_count(p) <= 12 and (ev.vector[FIELD_INDEX["a1_code_block"]] or ev.vector[FIELD_INDEX["a2_error_trace"]]):
        ev.set("a7_small_snippet", 1.0, "small isolated code/error snippet")

    # Contribution/design evidence.
    if _hit(r"\b(bước \d+|step \d+|pseudocode|thuật toán|algorithm|đầu tiên|sau đó|cuối cùng)\b", lower):
        ev.set("d1_algorithm", 1.0, "student algorithm/steps")
    if _hit(r"\b(score|formula|công thức|tính theo|weighted|trọng số|=\s*\d+(\.\d+)?\s*[*x×])\b", lower):
        ev.set("d2_computation_logic", 1.0, "formula/computation logic")
    if _hit(r"\b(class|entity|model|interface|module|service|repository|controller|schema|relationship|architecture)\b", lower):
        ev.set("d3_structure", 1.0, "structure/class/entity design")
    if _hit(r"\b(flow|pipeline|luồng|request\s*->|a\s*->\s*b|input\s*->\s*process|data flow)\b", lower):
        ev.set("d4_data_flow", 1.0, "data/process flow")
    if _hit(r"\b(file|hàm|function|class|line|dòng|module|component|endpoint)\s+[\w./#:-]+", lower):
        ev.set("d5_location", 1.0, "specific location")
    if _hit(r"\b(em nghĩ|i think|có thể do|probably because|nguyên nhân là|do .* gây ra)\b", lower):
        ev.set("d6_hypothesis", 1.0, "hypothesized cause")
    if _hit(r"\b(test case|unit test|expected|input|output|assert|given|when|then|kết quả mong đợi)\b", lower):
        ev.set("d7_test_cases", 1.0, "test/expected I/O")

    # Request/expectation evidence.
    request_patterns = [
        ("r1_create", r"\b(viết|tạo|build|generate|create|make|code cho|làm app|from scratch|từ đầu)\b"),
        ("r2_fix", r"\b(fix|sửa lỗi|debug|khắc phục|resolve|chữa lỗi|bị sao)\b"),
        ("r3_add_feature", r"\b(thêm|add|extend|bổ sung|new feature|mở rộng)\b"),
        ("r4_refactor", r"\b(refactor|optimize|clean up|restructure|format|cải thiện code)\b"),
        ("r5_convert", r"\b(convert|translate|migrate|đổi sang|chuyển sang)\b"),
        ("r6_explain", r"\b(tại sao|vì sao|why|how does|giải thích|explain|cơ chế|nguyên lý)\b"),
        ("r7_implement_design", r"\b(theo thiết kế|theo thuật toán|implement theo|dựa trên flow|based on my design|tôi thiết kế)\b"),
        ("r8_review", r"\b(review|đúng chưa|check|kiểm tra|edge case|security|bảo mật|complexity|evaluate)\b"),
        ("r9_generate_tests", r"\b(unit test|test case|viết test|generate tests|assertion)\b"),
        ("r10_syntax_lookup", r"\b(cú pháp|syntax|lệnh|command|api|tham số|parameter|regex|signature|flag)\b"),
    ]
    for field, pat in request_patterns:
        h = _hit(pat, lower)
        if h:
            ev.set(field, 1.0, h)

    # Output-side evidence.  Rules only identify obvious forms/mismatches; LLM should judge subtle alignment.
    if o.strip():
        ev.set("output_present", 1.0, "AI output present")
        out_lower = o.lower()
        if _hit(r"```|\b(def|class|function|const|let|var|public|private|import|return)\b", o):
            ev.set("output_direct_code_patch", 1.0, "code appears in output")
        if _line_count(o) >= 70 or _hit(r"\b(full implementation|complete solution|entire program|toàn bộ code|full code)\b", out_lower):
            ev.set("output_complete_solution", 1.0, "large/full solution output")
        if _hit(r"\b(because|vì|do đó|nguyên nhân|cơ chế|why|how|giải thích|lý do)\b", out_lower):
            ev.set("output_explanation", 1.0, "explanatory output")
        if _hit(r"\b(review|edge case|security|complexity|đúng|sai|vấn đề|recommend|rủi ro)\b", out_lower):
            ev.set("output_review_feedback", 1.0, "review-like output")
        if _hit(r"\b(test|assert|expected|pytest|unittest|jest|xunit|input|output)\b", out_lower):
            ev.set("output_tests", 1.0, "tests in output")
        if _line_count(o) <= 15 and _hit(r"\b(syntax|command|api|parameter|regex|cú pháp|lệnh)\b", out_lower):
            ev.set("output_narrow_reference", 1.0, "narrow reference output")

        # Obvious mismatch heuristics based on prompt request vs output form.
        if ev.vector[FIELD_INDEX["r10_syntax_lookup"]] >= 0.5 and ev.vector[FIELD_INDEX["output_complete_solution"]] >= 0.5:
            ev.set("over_scope_broader", 1.0, "syntax lookup received complete solution")
            ev.set("role_escalation", 1.0, "lookup role escalated")
        if ev.vector[FIELD_INDEX["r6_explain"]] >= 0.5 and ev.vector[FIELD_INDEX["output_direct_code_patch"]] >= 0.5 and ev.vector[FIELD_INDEX["output_explanation"]] < 0.5:
            ev.set("form_or_pedagogy_mismatch", 1.0, "explanation request received mostly code")
        if ev.vector[FIELD_INDEX["r8_review"]] >= 0.5 and ev.vector[FIELD_INDEX["output_complete_solution"]] >= 0.5:
            ev.set("agency_takeover", 1.0, "review request received full rewrite")

    return ev


def merge_rule_and_llm_vectors(llm_values: Sequence[float], rule: RuleEvidence) -> list[float]:
    """Merge deterministic rule evidence with LLM features.

    Rule evidence is a lower-bound override: if a regex sees code/error/action, that
    feature must be active.  Non-hit rule zeros do not erase LLM semantic judgments.
    Legacy 26D LLM vectors are padded before merging.
    """
    values = [float(v) for v in llm_values]
    if len(values) == PROMPT_VECTOR_SIZE:
        values = values + [0.0] * (VECTOR_SIZE - PROMPT_VECTOR_SIZE)
    if len(values) != VECTOR_SIZE:
        raise ValueError(f"Expected {PROMPT_VECTOR_SIZE} or {VECTOR_SIZE} LLM values, got {len(llm_values)}")
    return [max(values[i], rule.vector[i]) for i in range(VECTOR_SIZE)]


def rule_evidence_as_dict(rule: RuleEvidence) -> dict[str, Any]:
    return {
        "rule_vector": [round(v, 4) for v in rule.vector],
        "hits": rule.hits,
    }
