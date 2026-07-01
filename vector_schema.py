"""Human-readable APC transaction vector schema for diagnostics."""

from __future__ import annotations

from typing import Any, Sequence


VECTOR_FIELDS = [
    ("p1", "is_coding", "STUDENT_PROMPT thuộc bài toán lập trình/coding/kỹ thuật."),
    ("p2", "has_context", "STUDENT_PROMPT đủ context để đánh giá, không thiếu ngữ cảnh/injection."),
    ("a1", "a1_code_block", "Prompt có code block, function, class, complete file, SQL script hoặc artifact code đáng kể."),
    ("a2", "a2_error_trace", "Prompt có compiler/runtime error, exception, stack trace, diagnostic hoặc error message rõ."),
    ("a3", "a3_sql_schema", "Prompt có SQL query, database schema, table definition, migration hoặc DB config."),
    ("a4", "a4_config_file", "Prompt có JSON/YAML/INI/XML/env/build/package/Docker/config content."),
    ("a5", "a5_system_log", "Prompt có terminal output, server log, command output, timestamped log hoặc execution trace."),
    ("a6", "a6_extensive_context", "Prompt có context lớn, nhiều file, whole project hoặc complete solution."),
    ("a7", "a7_small_snippet", "Prompt chỉ có một snippet/statement/artifact nhỏ, cô lập."),
    ("d1", "d1_algorithm", "SV cung cấp algorithm steps, pseudocode, numbered flow hoặc procedural logic."),
    ("d2", "d2_computation_logic", "SV định nghĩa formula, transformation, scoring rule hoặc computation method."),
    ("d3", "d3_structure", "SV cung cấp class/entity/module architecture, schema design, relationships hoặc design pattern."),
    ("d4", "d4_data_flow", "SV mô tả data/request/pipeline/protocol flow hoặc transformation A -> B -> C."),
    ("d5", "d5_location", "SV chỉ ra file/class/function/line/module/location cụ thể cần sửa/inspect."),
    ("d6", "d6_hypothesis", "SV đưa ra nguyên nhân/hypothesis và vẫn yêu cầu AI fix/debug."),
    ("d7", "d7_test_cases", "SV cung cấp test input/output, expected result, validation example hoặc test cases."),
    ("r1", "r1_create", "Prompt yêu cầu create/write/build/generate solution/app/file/program mới từ đầu."),
    ("r2", "r2_fix", "Prompt yêu cầu fix/debug/resolve error, bug, crash hoặc malfunction."),
    ("r3", "r3_add_feature", "Prompt yêu cầu add/extend feature trong logic/artifact hiện có."),
    ("r4", "r4_refactor", "Prompt yêu cầu refactor/optimize/clean/format/restructure/improve code."),
    ("r5", "r5_convert", "Prompt yêu cầu convert/migrate/translate code/config sang language/framework/format khác."),
    ("r6", "r6_explain", "Prompt yêu cầu explain/teach/clarify why/how concept, code, bug hoặc behavior hoạt động."),
    ("r7", "r7_implement_design", "Prompt yêu cầu implement theo algorithm/design/formula/flow do SV cung cấp."),
    ("r8", "r8_review", "Prompt yêu cầu review/evaluate/check correctness/security/edge cases/complexity/quality."),
    ("r9", "r9_generate_tests", "Prompt yêu cầu viết unit tests/integration tests/test cases/assertions."),
    ("r10", "r10_syntax_lookup", "Prompt hỏi reference/syntax/API/command/parameter/regex/function signature/tool option rất hẹp."),
    ("o1", "output_present", "Có AI_OUTPUT để đánh giá transaction Prompt + Output."),
    ("o2", "role_aligned", "AI_OUTPUT đúng vai trò mà STUDENT_PROMPT kỳ vọng."),
    ("o3", "scope_aligned", "AI_OUTPUT giữ đúng phạm vi yêu cầu, không mở rộng quá mức."),
    ("o4", "agency_aligned", "AI_OUTPUT không tước quyền chủ động/đóng góp của SV so với kỳ vọng prompt."),
    ("o5", "form_aligned", "AI_OUTPUT đúng dạng trả lời mong đợi: giải thích/code patch/review/test/syntax."),
    ("o6", "pedagogy_aligned", "AI_OUTPUT có mức minh bạch học tập phù hợp với level/kỳ vọng."),
    ("o7", "output_asks_clarification", "AI_OUTPUT hỏi lại để làm rõ thay vì tự làm khi thiếu ngữ cảnh."),
    ("o8", "output_complete_solution", "AI_OUTPUT chứa complete solution/full program/full module/full implementation."),
    ("o9", "output_direct_code_patch", "AI_OUTPUT chứa edited code/patch/replacement function/direct code changes."),
    ("o10", "output_explanation", "AI_OUTPUT giải thích concept/mechanism/cause/trade-off/why-how."),
    ("o11", "output_review_feedback", "AI_OUTPUT review/evaluate/check edge cases/security/quality."),
    ("o12", "output_tests", "AI_OUTPUT cung cấp unit tests/integration tests/test cases/assertions/expected I/O."),
    ("o13", "output_narrow_reference", "AI_OUTPUT là lookup hẹp: syntax/command/API signature/parameter/regex."),
    ("m1", "over_scope_broader", "AI_OUTPUT rộng hơn prompt: full solution/module/architecture/implementation ngoài yêu cầu."),
    ("m2", "under_answer_missing", "AI_OUTPUT không trả lời trọng tâm, bỏ thiếu yêu cầu chính hoặc quá mơ hồ."),
    ("m3", "role_escalation", "AI_OUTPUT nâng vai trò AI lên mức ủy thác cao hơn prompt kỳ vọng."),
    ("m4", "agency_takeover", "AI_OUTPUT làm thay phần lẽ ra thuộc quyền chủ động/đóng góp của SV."),
    ("m5", "form_or_pedagogy_mismatch", "AI_OUTPUT sai dạng hoặc thiếu minh bạch học tập so với prompt."),
]


def vector_detail(values: Sequence[float | int] | None) -> list[dict[str, Any]]:
    if values is None:
        values = []
    normalized = [float(v) for v in values]
    rows: list[dict[str, Any]] = []
    for index, (key, name, description) in enumerate(VECTOR_FIELDS):
        value = normalized[index] if index < len(normalized) else 0.0
        rows.append(
            {
                "index": index,
                "key": key,
                "name": name,
                "value": round(value, 4),
                "active": value >= 0.5,
                "description": description,
            }
        )
    return rows
