"""Human-readable transaction vector schema for Prompt A/E/C + Output 6D.

Vector v4 is grounded in the two rubric sheets and the formula document:
- Prompt side: Artifact / Expectation / Contribution.
- Output side: Role / Scope / Agency / Form / Pedagogy / Mismatch.
"""

from __future__ import annotations

from typing import Any, Sequence


VECTOR_FIELDS = [
    # Prefix / safety / reject-option
    ("p1", "is_coding", "Prompt belongs to programming/coding/technical learning."),
    ("p2", "has_context", "Prompt has enough context to evaluate and is not empty."),
    ("p3", "non_coding", "Prompt is outside coding/technical scope."),
    ("p4", "ambiguous", "Prompt intent is unclear or insufficient to classify."),
    ("p5", "bypass", "Prompt contains self-label, rubric bypass, or injection-like instruction."),

    # Prompt Artifact A
    ("a1", "spec_or_vibe_only", "Only business requirement/spec/user story/vibe-code in natural language; WHAT but no HOW."),
    ("a2", "existing_code_or_snippet", "Existing code/function/file/component/query artifact is provided."),
    ("a3", "error_log_or_testfail", "Error message, stack trace, terminal log, or failed test is provided."),
    ("a4", "config_sql_terminal_ide_small", "Small config key, SQL command/query, terminal command/output, IDE/environment item."),
    ("a5", "full_solution_or_module_owned", "Student owns a complete/near-complete solution/module for checking."),
    ("a6", "test_or_expected_behavior", "Test case, expected I/O, edge case, or pass/fail criterion is provided."),
    ("a7", "pseudocode_flow_code_how", "Student provides pseudocode, function-level flow, algorithm, or clear HOW."),

    # Prompt Expectation E
    ("e1", "create_or_add_feature_module", "Student wants AI to create from scratch or add a feature/module/project."),
    ("e2", "fix_optimize_refactor_existing", "Student wants AI to fix/debug/optimize/refactor existing artifact."),
    ("e3", "explain_direction_theory_example", "Student wants direction, explanation, theory with example, or code explanation."),
    ("e4", "implement_my_pseudocode", "Student wants AI to implement the provided pseudocode/flow/HOW."),
    ("e5", "review_yesno_challenge", "Student wants review, yes/no verification, challenge, critique, or edge-case checking."),
    ("e6", "minimal_lookup", "Student asks one small lookup: short theory, syntax/API, config, SQL, terminal, IDE."),

    # Prompt Contribution C
    ("c1", "what_only", "Student only contributes WHAT/spec/business requirement; no HOW."),
    ("c2", "localized_context", "Student contributes local code/context for a task."),
    ("c3", "debug_context_or_hypothesis", "Student gives bug location, hypothesis, error/log, or test failure context."),
    ("c4", "student_how_flow", "Student contributes pseudocode/function flow/algorithm/HOW."),
    ("c5", "student_solution_or_claim", "Student contributes own solution, answer, claim, or decision for review/challenge."),
    ("c6", "exact_small_need", "Student knows the exact small item they need to look up."),

    # Output presence and Role OR
    ("o0", "output_present", "AI_OUTPUT is present for transaction-level assessment."),
    ("or1", "out_full_build", "AI builds full code/feature/module/app or substantial new implementation."),
    ("or2", "out_fix_optimize_refactor", "AI fixes/debugs/optimizes/refactors an existing artifact."),
    ("or3", "out_explain_example", "AI explains, gives direction, theory, or examples."),
    ("or4", "out_implement_pseudocode", "AI implements according to student pseudocode/flow/HOW."),
    ("or5", "out_review_yesno_challenge", "AI reviews, gives yes/no with reasons, critiques, checks edge cases."),
    ("or6", "out_minimal_lookup", "AI answers narrowly: short theory/syntax/config/SQL/terminal/IDE."),

    # Scope OS
    ("os1", "scope_minimal", "Output is minimal: one concept/definition/command/1-3 lines."),
    ("os2", "scope_local", "Output is local: small snippet/example/function/patch."),
    ("os3", "scope_module", "Output covers a feature/module."),
    ("os4", "scope_full_system", "Output covers whole app/project/full solution."),

    # Agency OG
    ("og1", "agency_ai_led", "AI decides the main solution/design itself."),
    ("og2", "agency_shared", "AI adds suggestions while partially preserving student intent."),
    ("og3", "agency_student_led", "AI preserves student design/claim/agency."),
    ("og4", "design_contamination", "AI adds or changes algorithm/architecture/design not requested."),
    ("og5", "replaced_student_solution", "AI rewrites/replaces student solution instead of review/verification."),

    # Form OF
    ("of1", "form_full_code", "Output form is full code/full implementation."),
    ("of2", "form_patch", "Output form is patch/fix/refactor instructions/code."),
    ("of3", "form_explanation_example", "Output form is explanation plus example/code example."),
    ("of4", "form_implementation", "Output form is implementation of supplied pseudocode/flow."),
    ("of5", "form_review_checklist", "Output form is review/checklist/yes-no/challenge/test suggestions."),
    ("of6", "form_command_short", "Output form is short command/syntax/config/SQL/terminal/IDE answer."),

    # Pedagogy OT
    ("ot1", "pedagogy_brief", "Output is brief and direct."),
    ("ot2", "pedagogy_reasoned", "Output gives reasons/explanation."),
    ("ot3", "pedagogy_diagnostic", "Output diagnoses causes, tradeoffs, or edge cases."),
    ("ot4", "pedagogy_code_only", "Output is mostly copy-paste code/answer with little explanation."),

    # Mismatch Delta
    ("dlt1", "role_shift", "Observed role differs from expected role."),
    ("dlt2", "scope_overreach", "Output scope is broader than expected scope."),
    ("dlt3", "agency_takeover", "AI takes over design/solution when not requested."),
    ("dlt4", "form_mismatch", "Output form differs from expected form."),
    ("dlt5", "over_answer", "Output gives more than needed."),
    ("dlt6", "under_answer", "Output misses the main request."),
    ("dlt7", "minimal_to_broad_shift", "L6 prompt became explanation/example/full code output."),
    ("dlt8", "explain_to_fix_shift", "L3 explanation prompt became fix/patch output."),
    ("dlt9", "review_replacement", "L5 review/yes-no prompt became rewrite/replacement."),
    ("dlt10", "design_contamination_delta", "L4 implement-my-flow prompt got AI-added design."),
]

FIELD_INDEX = {name: idx for idx, (_, name, _) in enumerate(VECTOR_FIELDS)}


def vector_detail(values: Sequence[float | int] | None) -> list[dict[str, Any]]:
    if values is None:
        values = []
    normalized = [float(v) for v in values]
    rows: list[dict[str, Any]] = []
    for index, (key, name, description) in enumerate(VECTOR_FIELDS):
        value = normalized[index] if index < len(normalized) else 0.0
        rows.append({
            "index": index,
            "key": key,
            "name": name,
            "value": round(value, 4),
            "active": value >= 0.5,
            "description": description,
        })
    return rows
