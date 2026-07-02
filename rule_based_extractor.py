"""Rule-based feature extraction for Prompt A/E/C + Output 6D vectors.

Rules are deterministic lower-bound evidence. The LLM extractor may still fill
semantic values, but obvious artifacts/verbs/output forms are anchored here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from apc_v4_engine import LEGACY_PROMPT_VECTOR_SIZE, LEGACY_TRANSACTION_VECTOR_SIZE, PROMPT_VECTOR_SIZE, VECTOR_SIZE
from vector_schema import FIELD_INDEX


@dataclass
class RuleEvidence:
    vector: list[float]
    hits: dict[str, list[str]] = field(default_factory=dict)

    def set(self, field_name: str, value: float = 1.0, evidence: str | None = None) -> None:
        idx = FIELD_INDEX[field_name]
        self.vector[idx] = max(self.vector[idx], float(value))
        if evidence:
            self.hits.setdefault(field_name, []).append(evidence[:180])


def _has(pattern: str, text: str, flags: int = re.IGNORECASE | re.MULTILINE) -> re.Match[str] | None:
    return re.search(pattern, text or "", flags)


def _hit(pattern: str, text: str, flags: int = re.IGNORECASE | re.MULTILINE) -> str | None:
    m = _has(pattern, text, flags)
    return m.group(0).strip() if m else None


def _line_count(text: str) -> int:
    return len([line for line in (text or "").splitlines() if line.strip()])


def _code_line_count(text: str) -> int:
    n = 0
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if _has(r"^(#include|using |import |from |def |class |function |const |let |var |public |private |protected |return\b|if\b|for\b|while\b|SELECT\b|INSERT\b|UPDATE\b|DELETE\b|CREATE\b)", line):
            n += 1
        elif _has(r"[{};]$|=>|\w+\s*=\s*.+", line):
            n += 1
    return n


def _has_code(text: str) -> bool:
    return bool(_has(r"```[\s\S]*?```", text) or _code_line_count(text) >= 2)


def _has_executable_code_example(text: str) -> bool:
    return bool(_has(r"```[\s\S]*?```", text) or _has(r"#include\s*<|int\s+main\s*\(|def\s+\w+\s*\(|class\s+\w+|function\s+\w+\s*\(", text) or _code_line_count(text) >= 2)


def extract_rule_vector(student_prompt: str, ai_output: str | None = None) -> RuleEvidence:
    v = [0.0] * VECTOR_SIZE
    ev = RuleEvidence(v)
    p = student_prompt or ""
    o = ai_output or ""
    lower = p.lower()

    if p.strip():
        ev.set("has_context", 1.0, "non-empty prompt")

    if _hit(r"\b(ignore previous|bỏ qua|hãy chấm L[0-6]|classify as L[0-6]|đừng theo rubric)\b", p):
        ev.set("bypass", 1.0, "self-label/bypass/injection-like phrase")

    coding_kw = r"\b(code|bug|debug|fix|error|exception|api|syntax|regex|sql|database|html|css|javascript|typescript|python|java|c#|c\+\+|printf|scanf|asp\.net|ef core|postgres|react|flutter|git|docker|terminal|vscode|ide|config|module|function|controller|service|component)\b"
    if _has(coding_kw, p) or _has_code(p):
        ev.set("is_coding", 1.0, "coding/technical keyword or code-like text")
    elif p.strip():
        ev.set("ambiguous", 0.5, "non-empty but no obvious coding evidence")

    # Artifact A.
    if _has_code(p) or _hit(r"\b(query|component|controller|service|repository|function|hàm|file)\b", lower):
        ev.set("existing_code_or_snippet", 1.0 if _has_code(p) else 0.5, "existing code/snippet/component evidence")
        ev.set("localized_context", 1.0 if _has_code(p) else 0.5, "localized artifact context")
    if _hit(r"\b(Traceback|Exception|Error:|TypeError|ValueError|NullReferenceException|IndexError|KeyError|SyntaxError|ReferenceError|stack trace|at line|line \d+|test failed|failed test|lỗi|bị lỗi)\b", p):
        val = 1.0 if _hit(r"\b(Traceback|Exception|Error:|stack trace|line \d+|test failed)\b", p) else 0.5
        ev.set("error_log_or_testfail", val, "error/log/test-failure evidence")
        ev.set("debug_context_or_hypothesis", val, "debug context")
    if _hit(r"\b(SELECT|INSERT INTO|UPDATE\s+\w+\s+SET|DELETE FROM|CREATE TABLE|ALTER TABLE|JOIN|SQL|DbContext|migration|appsettings\.json|\.env|docker-compose|package\.json|terminal|command|npm |git |vscode|IDE|config|connection string)\b", p):
        ev.set("config_sql_terminal_ide_small", 1.0, "config/sql/terminal/IDE small artifact")
    if _line_count(p) >= 80 or len(p) >= 5000 or _hit(r"\b(full project|whole project|toàn bộ project|complete solution|entire file|bài em làm|solution của em|module của em)\b", lower):
        ev.set("full_solution_or_module_owned", 1.0, "full/near-complete solution/module owned by student")
        ev.set("student_solution_or_claim", 1.0, "student solution/claim for verification")
    if _hit(r"\b(test case|unit test|expected|input|output|assert|given|when|then|edge case|kết quả mong đợi|pass/fail)\b", lower):
        ev.set("test_or_expected_behavior", 1.0, "test/expected behavior evidence")
    if _hit(r"\b(pseudocode|pseudo code|thuật toán|algorithm|flow|flow code|luồng|luồng code|bước \d+|step \d+|đầu tiên|sau đó|cuối cùng|validate\s*[-→>]+|hash\s*[-→>]+|save\s*[-→>]+)\b", lower):
        ev.set("pseudocode_flow_code_how", 1.0, "pseudocode/function-level flow/HOW")
        ev.set("student_how_flow", 1.0, "student supplied HOW/flow")

    # Expectation E.
    def req(field: str, pat: str, evidence: str):
        h = _hit(pat, lower)
        if h:
            ev.set(field, 1.0, h or evidence)

    req("create_or_add_feature_module", r"\b(viết|tạo|build|generate|create|make|code cho|làm app|from scratch|từ đầu|thêm tính năng|thêm module|add feature|new feature|module mới|chức năng mới)\b", "create/add feature/module")
    req("fix_optimize_refactor_existing", r"\b(fix|sửa lỗi|debug|khắc phục|resolve|chữa lỗi|bị sao|tối ưu|optimize|refactor|clean up|restructure|cải thiện code)\b", "fix/optimize/refactor")
    req("explain_direction_theory_example", r"\b(tại sao|vì sao|why|how does|giải thích|explain|cơ chế|nguyên lý|hướng đi|nên làm thế nào|cho ví dụ|example|giải thích code|lý thuyết)\b", "explain/direction/theory/example")
    req("implement_my_pseudocode", r"\b(theo pseudo|theo pseudocode|theo flow|theo thiết kế|theo thuật toán|implement theo|dựa trên flow|based on my design|tôi thiết kế|em có flow)\b", "implement my pseudocode/flow")
    req("review_yesno_challenge", r"\b(review|đúng chưa|đúng không|sai không|yes/no|check|kiểm tra|phản biện|challenge|edge case|security|bảo mật|complexity|evaluate)\b", "review/yes-no/challenge")
    # Minimal lookup: short theory, syntax, config, SQL, terminal, IDE. Includes bare "what is X" when not asking why/example.
    if _hit(r"\b(cú pháp|syntax|lệnh|command|api|tham số|parameter|regex|signature|flag|config|sql command|terminal|vscode|ide|what is|là gì)\b", lower):
        if not ev.vector[FIELD_INDEX["explain_direction_theory_example"]] or _hit(r"\b(what is|là gì)\b", lower):
            ev.set("minimal_lookup", 1.0, "minimal lookup / short theory / syntax / command")
            ev.set("exact_small_need", 1.0, "exact small lookup need")

    # Minimal command lookup like "lệnh git tạo branch" contains verb "tạo" but should be L6, not L1.
    if ev.vector[FIELD_INDEX["minimal_lookup"]] >= 0.5 and _hit(r"\b(lệnh|command|git|terminal|sql|cú pháp|syntax|api|tham số|parameter)\b", lower):
        if not _hit(r"\b(app|project|module mới|feature|chức năng|full|toàn bộ|code cho)\b", lower):
            ev.vector[FIELD_INDEX["create_or_add_feature_module"]] = 0.0

    # Vibe/spec only and WHAT-only.
    has_artifact = max(ev.vector[FIELD_INDEX[n]] for n in ["existing_code_or_snippet", "error_log_or_testfail", "config_sql_terminal_ide_small", "full_solution_or_module_owned", "test_or_expected_behavior", "pseudocode_flow_code_how"])
    if p.strip() and has_artifact < 0.5 and ev.vector[FIELD_INDEX["minimal_lookup"]] < 0.5:
        ev.set("spec_or_vibe_only", 1.0, "natural-language spec/vibe code; no technical artifact/HOW")
        ev.set("what_only", 1.0, "WHAT-only contribution")
    if ev.vector[FIELD_INDEX["create_or_add_feature_module"]] >= 0.5 and ev.vector[FIELD_INDEX["pseudocode_flow_code_how"]] < 0.5:
        ev.set("spec_or_vibe_only", 1.0, "create request without HOW => L1/vibe code")
        ev.set("what_only", 1.0, "WHAT-only create request")

    if _hit(r"\b(em nghĩ|i think|có thể do|probably because|nguyên nhân là|do .* gây ra|lỗi ở|ở hàm|ở dòng)\b", lower):
        ev.set("debug_context_or_hypothesis", 1.0, "debug hypothesis/location")
    if ev.vector[FIELD_INDEX["review_yesno_challenge"]] >= 0.5 and max(ev.vector[FIELD_INDEX["full_solution_or_module_owned"]], ev.vector[FIELD_INDEX["test_or_expected_behavior"]]) >= 0.5:
        ev.set("student_solution_or_claim", 1.0, "student has artifact/claim for review")

    # Output 6D.
    if o.strip():
        ev.set("output_present", 1.0, "AI output present")
        out_lower = o.lower()
        code_lines = _code_line_count(o)
        has_exec_code = _has_executable_code_example(o)
        line_count = _line_count(o)

        if _hit(r"\b(could you clarify|can you provide|bạn muốn|cần thêm|hãy gửi|chưa đủ thông tin|clarify)\b", out_lower):
            ev.set("under_answer", 0.5, "AI asks for clarification / incomplete")
        if line_count <= 5:
            ev.set("scope_minimal", 1.0, "short/minimal output")
        elif line_count <= 25:
            ev.set("scope_local", 1.0, "local output length")
        elif line_count <= 80:
            ev.set("scope_module", 0.75, "module-sized output")
        else:
            ev.set("scope_full_system", 1.0, "full-system-sized output")

        if _hit(r"\b(full implementation|complete solution|entire program|toàn bộ code|full code|full project|module hoàn chỉnh)\b", out_lower) or line_count >= 80:
            ev.set("out_full_build", 1.0, "full build output")
            ev.set("form_full_code", 1.0, "full code form")
            ev.set("agency_ai_led", 1.0, "AI-led full build")
            ev.set("scope_full_system", 1.0, "full-system scope")
        if _hit(r"\b(here(?:'s| is) the fix|replace with|use this code|patch below|sửa thành|đổi thành|fixed code|refactor)\b", out_lower) or (has_exec_code and ev.vector[FIELD_INDEX["fix_optimize_refactor_existing"]] >= 0.5):
            ev.set("out_fix_optimize_refactor", 1.0, "fix/patch/refactor output")
            ev.set("form_patch", 1.0, "patch form")
            ev.set("scope_local", 1.0, "local patch scope")
        if _hit(r"\b(because|vì|do đó|nguyên nhân|cơ chế|why|how|giải thích|lý do|means|used to|comes from|format)\b", out_lower):
            ev.set("out_explain_example", 1.0, "explanation/theory output")
            ev.set("pedagogy_reasoned", 1.0, "reasoned explanation")
            if has_exec_code or _hit(r"\b(example|ví dụ)\b", out_lower):
                ev.set("form_explanation_example", 1.0, "explanation plus example/code example")
                ev.set("scope_local", max(ev.vector[FIELD_INDEX["scope_local"]], 0.75), "example expands scope to local")
        if _hit(r"\b(theo flow|theo pseudocode|theo thiết kế|implements your flow|based on your flow)\b", out_lower):
            ev.set("out_implement_pseudocode", 1.0, "implementation of student pseudocode")
            ev.set("form_implementation", 1.0, "implementation form")
            ev.set("agency_student_led", 1.0, "student-led design preserved")
        if _hit(r"\b(review|edge case|security|complexity|đúng|sai|vấn đề|recommend|rủi ro|phản biện|checklist|yes)\b", out_lower):
            ev.set("out_review_yesno_challenge", 1.0, "review/yes-no/challenge output")
            ev.set("form_review_checklist", 1.0, "review/checklist form")
            ev.set("pedagogy_diagnostic", 1.0, "diagnostic review")
        if line_count <= 8 and _hit(r"\b(syntax|command|api|parameter|regex|cú pháp|lệnh|config|sql|terminal|vscode|ide|printf\s*\(|%d)\b", out_lower) and not has_exec_code:
            ev.set("out_minimal_lookup", 1.0, "minimal lookup output")
            ev.set("form_command_short", 1.0, "short command/syntax form")
            ev.set("scope_minimal", 1.0, "minimal scope")
            ev.set("pedagogy_brief", 1.0, "brief direct output")

        if has_exec_code and ev.vector[FIELD_INDEX["out_full_build"]] < 0.5 and ev.vector[FIELD_INDEX["out_fix_optimize_refactor"]] < 0.5:
            ev.set("out_explain_example", max(ev.vector[FIELD_INDEX["out_explain_example"]], 0.75), "code example in explanatory output")
            ev.set("form_explanation_example", 1.0, "code example form")
            ev.set("scope_local", max(ev.vector[FIELD_INDEX["scope_local"]], 0.75), "code example local scope")

        if has_exec_code and max(ev.vector[FIELD_INDEX["pedagogy_reasoned"]], ev.vector[FIELD_INDEX["pedagogy_diagnostic"]]) < 0.5:
            ev.set("pedagogy_code_only", 1.0, "mostly code-only/copy-paste output")
        else:
            ev.set("pedagogy_brief", max(ev.vector[FIELD_INDEX["pedagogy_brief"]], 0.5), "some textual support")

        # Mismatch heuristics from prompt expectation vs output behavior.
        if ev.vector[FIELD_INDEX["minimal_lookup"]] >= 0.5 and max(ev.vector[FIELD_INDEX["out_explain_example"]], ev.vector[FIELD_INDEX["scope_local"]], ev.vector[FIELD_INDEX["scope_module"]], ev.vector[FIELD_INDEX["scope_full_system"]], ev.vector[FIELD_INDEX["form_explanation_example"]], ev.vector[FIELD_INDEX["form_full_code"]]) >= 0.5:
            ev.set("minimal_to_broad_shift", 1.0, "L6 prompt received explanation/example/full code")
            ev.set("scope_overreach", 1.0, "minimal prompt received broader output")
            ev.set("form_mismatch", 0.75, "minimal expected form expanded")
            ev.set("over_answer", 1.0, "more than requested")
        if ev.vector[FIELD_INDEX["explain_direction_theory_example"]] >= 0.5 and ev.vector[FIELD_INDEX["out_fix_optimize_refactor"]] >= 0.5:
            ev.set("explain_to_fix_shift", 1.0, "explain prompt received fix")
            ev.set("role_shift", 1.0, "role shifted from explain to fix")
        if ev.vector[FIELD_INDEX["review_yesno_challenge"]] >= 0.5 and (ev.vector[FIELD_INDEX["out_full_build"]] >= 0.5 or ev.vector[FIELD_INDEX["replaced_student_solution"]] >= 0.5):
            ev.set("review_replacement", 1.0, "review prompt received replacement")
            ev.set("agency_takeover", 1.0, "AI replaced student solution")
        if ev.vector[FIELD_INDEX["implement_my_pseudocode"]] >= 0.5 and ev.vector[FIELD_INDEX["design_contamination"]] >= 0.5:
            ev.set("design_contamination_delta", 1.0, "L4 design contaminated by AI")

    return ev


def _legacy_to_new(values: Sequence[float]) -> list[float]:
    from apc_v4_engine import _legacy_to_new as engine_legacy_to_new  # lazy import avoids cycle at module import time
    return engine_legacy_to_new(values)


def merge_rule_and_llm_vectors(llm_values: Sequence[float], rule: RuleEvidence) -> list[float]:
    values = [float(v) for v in llm_values]
    if len(values) == VECTOR_SIZE:
        base = values
    elif len(values) in (PROMPT_VECTOR_SIZE, LEGACY_PROMPT_VECTOR_SIZE, LEGACY_TRANSACTION_VECTOR_SIZE):
        base = _legacy_to_new(values)
    else:
        raise ValueError(f"Expected vector length {PROMPT_VECTOR_SIZE}, {LEGACY_PROMPT_VECTOR_SIZE}, {LEGACY_TRANSACTION_VECTOR_SIZE}, or {VECTOR_SIZE}; got {len(values)}")
    return [max(base[i], rule.vector[i]) for i in range(VECTOR_SIZE)]


def rule_evidence_as_dict(rule: RuleEvidence) -> dict[str, Any]:
    return {"rule_vector": [round(v, 4) for v in rule.vector], "hits": rule.hits}
