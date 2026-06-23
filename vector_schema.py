"""Human-readable APC vector schema for diagnostics."""

from __future__ import annotations

from typing import Any, Sequence


VECTOR_FIELDS = [
    ("p1", "is_coding", "Prompt thuộc bài toán lập trình/coding."),
    ("p2", "has_context", "Prompt đủ context để đánh giá, không phải thiếu ngữ cảnh/injection."),
    ("a1", "a1_code_block", "Có code block, function, class, complete file, SQL script hoặc artifact code đáng kể."),
    ("a2", "a2_error_trace", "Có compiler/runtime error, exception, stack trace, diagnostic hoặc error message rõ."),
    ("a3", "a3_sql_schema", "Có SQL query, database schema, table definition, migration hoặc DB config."),
    ("a4", "a4_config_file", "Có JSON/YAML/INI/XML/env/build/package/Docker/config content."),
    ("a5", "a5_system_log", "Có terminal output, server log, command output, timestamped log hoặc execution trace."),
    ("a6", "a6_extensive_context", "Có context lớn, nhiều file, whole project hoặc complete solution."),
    ("a7", "a7_small_snippet", "Chỉ có một snippet/statement/artifact nhỏ, cô lập."),
    ("d1", "d1_algorithm", "Student cung cấp algorithm steps, pseudocode, numbered flow hoặc procedural logic."),
    ("d2", "d2_computation_logic", "Student định nghĩa formula, transformation, scoring rule hoặc computation method."),
    ("d3", "d3_structure", "Student cung cấp class/entity/module architecture, schema design, relationships hoặc design pattern."),
    ("d4", "d4_data_flow", "Student mô tả data/request/pipeline/protocol flow hoặc transformation A -> B -> C."),
    ("d5", "d5_location", "Student chỉ ra file/class/function/line/module/location cụ thể cần sửa/inspect."),
    ("d6", "d6_hypothesis", "Student đưa ra nguyên nhân/hypothesis và vẫn yêu cầu AI fix/debug."),
    ("d7", "d7_test_cases", "Student cung cấp test input/output, expected result, validation example hoặc test cases."),
    ("r1", "r1_create", "Yêu cầu create/write/build/generate solution/app/file/program mới từ đầu."),
    ("r2", "r2_fix", "Yêu cầu fix/debug/resolve error, bug, crash hoặc malfunction."),
    ("r3", "r3_add_feature", "Yêu cầu add/extend feature trong logic/artifact hiện có."),
    ("r4", "r4_refactor", "Yêu cầu refactor/optimize/clean/format/restructure/improve code."),
    ("r5", "r5_convert", "Yêu cầu convert/migrate/translate code/config sang language/framework/format khác."),
    ("r6", "r6_explain", "Yêu cầu explain/teach/clarify why/how concept, code, bug hoặc behavior hoạt động."),
    ("r7", "r7_implement_design", "Yêu cầu implement theo algorithm/design/formula/flow do student cung cấp."),
    ("r8", "r8_review", "Yêu cầu review/evaluate/check correctness/security/edge cases/complexity/quality."),
    ("r9", "r9_generate_tests", "Yêu cầu viết unit tests/integration tests/test cases/assertions."),
    ("r10", "r10_syntax_lookup", "Hỏi reference/syntax/API/command/parameter/regex/function signature/tool option rất hẹp."),
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
