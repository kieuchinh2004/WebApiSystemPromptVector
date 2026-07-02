"""Deterministic scoring engine for Prompt A/E/C + Output 6D vector model.

This version follows the user's two rubric sheets and the v4 formula document:
- Prompt Candidate Level Lp is computed only from Artifact / Expectation / Contribution.
- Output Observed Level Lo is computed independently from 6D output evidence:
  Role / Scope / Agency / Form / Pedagogy / Mismatch.
- If output deviates from the prompt expectation, the system keeps Lp and warns by Lo.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from rubric_schema import RUBRIC_LEVELS
from vector_schema import FIELD_INDEX, VECTOR_FIELDS

PROMPT_VECTOR_SIZE = 24   # p1..p5 + a1..a7 + e1..e6 + c1..c6
LEGACY_PROMPT_VECTOR_SIZE = 26
LEGACY_TRANSACTION_VECTOR_SIZE = 44
VECTOR_SIZE = len(VECTOR_FIELDS)  # 60

TAU = 8.0
ACCEPT_CONFIDENCE = 0.60
ACCEPT_MARGIN = 0.15
GATING_THRESHOLD = 0.50
STABLE_FIT_THRESHOLD = 0.70
WARNING_THRESHOLD = 0.30
CONFIRM_THRESHOLD = 0.60

LEVELS = tuple(f"L{i}" for i in range(1, 7))
CONTEXT_LEVEL = "Thieu context"
CONFIRMATION_LEVEL = "Can hoi lai SV"


def _idx(name: str) -> int:
    return FIELD_INDEX[name]


def _blank() -> list[float]:
    return [0.0] * VECTOR_SIZE


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def _legacy_to_new(values: Sequence[float]) -> list[float]:
    """Map the old 26D/44D vectors into the new 60D schema.

    This keeps backward compatibility with previous tests/API clients while making the
    new vector layout the canonical one.
    """
    old = list(values)
    x = _blank()
    x[_idx("is_coding")] = old[0] if len(old) > 0 else 0.0
    x[_idx("has_context")] = old[1] if len(old) > 1 else 0.0

    if len(old) < LEGACY_PROMPT_VECTOR_SIZE:
        return x

    # Old prompt features: prefix + a1..a7 + d1..d7 + r1..r10.
    oa = old[2:9]
    od = old[9:16]
    orq = old[16:26]

    # Artifact/contribution mapping.
    if max(oa[0], oa[6]) >= 0.5:
        x[_idx("existing_code_or_snippet")] = max(oa[0], oa[6])
        x[_idx("localized_context")] = max(x[_idx("localized_context")], max(oa[0], oa[6]))
    x[_idx("error_log_or_testfail")] = max(oa[1], oa[4])
    x[_idx("config_sql_terminal_ide_small")] = max(oa[2], oa[3], oa[4])
    x[_idx("full_solution_or_module_owned")] = oa[5]
    x[_idx("test_or_expected_behavior")] = od[6]
    x[_idx("pseudocode_flow_code_how")] = max(od[0], od[1], od[2], od[3])
    x[_idx("student_how_flow")] = max(od[0], od[1], od[2], od[3])
    x[_idx("debug_context_or_hypothesis")] = max(od[4], od[5], oa[1], oa[4])
    x[_idx("student_solution_or_claim")] = max(oa[5], od[6])

    # Expectation mapping.
    x[_idx("create_or_add_feature_module")] = max(orq[0], orq[2])
    x[_idx("fix_optimize_refactor_existing")] = max(orq[1], orq[3], orq[4])
    x[_idx("explain_direction_theory_example")] = orq[5]
    x[_idx("implement_my_pseudocode")] = orq[6]
    x[_idx("review_yesno_challenge")] = max(orq[7], orq[8])
    x[_idx("minimal_lookup")] = orq[9]

    # WHAT-only/spec inferred when create request has no code/HOW.
    if x[_idx("create_or_add_feature_module")] >= 0.5 and max(x[_idx("existing_code_or_snippet")], x[_idx("pseudocode_flow_code_how")]) < 0.5:
        x[_idx("spec_or_vibe_only")] = 1.0
        x[_idx("what_only")] = 1.0
    if x[_idx("minimal_lookup")] >= 0.5:
        x[_idx("exact_small_need")] = 1.0

    if len(old) >= LEGACY_TRANSACTION_VECTOR_SIZE:
        x[_idx("output_present")] = old[26]
        role_aligned, scope_aligned, agency_aligned, form_aligned, pedagogy_aligned = old[27:32]
        x[_idx("out_full_build")] = old[33]
        x[_idx("out_fix_optimize_refactor")] = old[34]
        x[_idx("out_explain_example")] = old[35]
        x[_idx("out_review_yesno_challenge")] = max(old[36], old[37])
        x[_idx("out_minimal_lookup")] = old[38]
        # Scope/form/agency rough mapping from old output-form evidence.
        x[_idx("scope_full_system")] = old[33]
        x[_idx("scope_local")] = max(old[34], old[35], old[36], old[37])
        x[_idx("scope_minimal")] = old[38]
        x[_idx("agency_ai_led")] = max(old[33], old[42])
        x[_idx("agency_student_led")] = agency_aligned
        x[_idx("replaced_student_solution")] = old[42]
        x[_idx("form_full_code")] = old[33]
        x[_idx("form_patch")] = old[34]
        x[_idx("form_explanation_example")] = old[35]
        x[_idx("form_review_checklist")] = max(old[36], old[37])
        x[_idx("form_command_short")] = old[38]
        x[_idx("pedagogy_reasoned")] = max(old[35], pedagogy_aligned)
        x[_idx("pedagogy_code_only")] = old[33]
        x[_idx("scope_overreach")] = old[39]
        x[_idx("under_answer")] = old[40]
        x[_idx("role_shift")] = old[41]
        x[_idx("agency_takeover")] = old[42]
        x[_idx("form_mismatch")] = old[43]
        x[_idx("over_answer")] = max(old[39], old[41])
        if x[_idx("minimal_lookup")] >= 0.5:
            x[_idx("minimal_to_broad_shift")] = max(old[33], old[35], old[39])
        if x[_idx("explain_direction_theory_example")] >= 0.5:
            x[_idx("explain_to_fix_shift")] = old[34]
        if x[_idx("review_yesno_challenge")] >= 0.5:
            x[_idx("review_replacement")] = old[42]
        if x[_idx("implement_my_pseudocode")] >= 0.5:
            x[_idx("design_contamination_delta")] = old[42]

    return x


def _parse_vector(vector_input: str | Sequence[float | int]) -> list[float]:
    if isinstance(vector_input, str):
        vector_str = "".join(vector_input.split())
        allowed_lengths = {LEGACY_PROMPT_VECTOR_SIZE, LEGACY_TRANSACTION_VECTOR_SIZE, PROMPT_VECTOR_SIZE, VECTOR_SIZE}
        if len(vector_str) not in allowed_lengths or not all(c in "01" for c in vector_str):
            raise ValueError(
                f"Vector string must contain exactly one of {sorted(allowed_lengths)} binary characters, "
                f"got {len(vector_str)}: {vector_str!r}"
            )
        values = [float(c) for c in vector_str]
    else:
        allowed_lengths = {LEGACY_PROMPT_VECTOR_SIZE, LEGACY_TRANSACTION_VECTOR_SIZE, PROMPT_VECTOR_SIZE, VECTOR_SIZE}
        if len(vector_input) not in allowed_lengths:
            raise ValueError(
                f"Vector list must contain exactly one of {sorted(allowed_lengths)} elements, got {len(vector_input)}"
            )
        values = [float(v) for v in vector_input]
        invalid = [v for v in values if not 0.0 <= v <= 1.0]
        if invalid:
            raise ValueError(f"Vector values must be in [0, 1], got invalid values: {invalid[:5]}")

    if len(values) == VECTOR_SIZE:
        return values
    if len(values) == PROMPT_VECTOR_SIZE:
        return values + [0.0] * (VECTOR_SIZE - PROMPT_VECTOR_SIZE)
    return _legacy_to_new(values)


def _binary_view(values: Sequence[float]) -> str:
    return "".join("1" if value >= GATING_THRESHOLD else "0" for value in values)


def _softmax(scores: Sequence[float], active: Sequence[bool] | None = None) -> list[float]:
    if active is None:
        active = [True] * len(scores)
    active_scores = [score for score, enabled in zip(scores, active) if enabled]
    if not active_scores:
        active_scores = list(scores)
        active = [True] * len(scores)
    max_logit = max(TAU * score for score in active_scores)
    exps = [math.exp(TAU * score - max_logit) if enabled else 0.0 for score, enabled in zip(scores, active)]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _top_margin(probs: Sequence[float]) -> float:
    vals = sorted(probs, reverse=True)
    return vals[0] - vals[1] if len(vals) > 1 else (vals[0] if vals else 0.0)


def _active(value: float) -> bool:
    return value >= GATING_THRESHOLD


def _field(x: Sequence[float], name: str) -> float:
    return float(x[_idx(name)])


def _role_index_from_prompt(x: Sequence[float], candidate_level: str) -> int:
    mapping = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6}
    return mapping.get(candidate_level, 0)


def _role_index_from_output(output_level: str | None) -> int:
    mapping = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6}
    return mapping.get(output_level or "", 0)


def _scope_expected(candidate_level: str) -> int:
    return {"L1": 3, "L2": 2, "L3": 1, "L4": 2, "L5": 1, "L6": 1}.get(candidate_level, 1)


def _scope_observed(x: Sequence[float]) -> int:
    vals = [
        (_field(x, "scope_minimal"), 1),
        (_field(x, "scope_local"), 2),
        (_field(x, "scope_module"), 3),
        (_field(x, "scope_full_system"), 4),
    ]
    return max(vals, key=lambda t: (t[0], t[1]))[1]


def _level_mismatch_warning(candidate_level: str, output_level: str | None) -> str | None:
    if output_level is None or output_level == candidate_level:
        return None
    output_rubric = RUBRIC_LEVELS.get(output_level)
    candidate_rubric = RUBRIC_LEVELS.get(candidate_level)
    if not output_rubric or not candidate_rubric:
        return None
    return (
        f"Warning {output_level}: AI_OUTPUT behaves like {output_level} ({output_rubric.name}) "
        f"while STUDENT_PROMPT candidate is {candidate_level} ({candidate_rubric.name})."
    )


def compute_apc_v4(vector_input: str | Sequence[float | int]) -> dict[str, Any]:
    x = [_clamp(v) for v in _parse_vector(vector_input)]
    vector_values = [round(value, 4) for value in x]
    vector_str = _binary_view(x)

    is_coding = _active(_field(x, "is_coding")) and not _active(_field(x, "non_coding"))
    has_context = _active(_field(x, "has_context")) and not _active(_field(x, "bypass"))

    # Prompt side fields.
    a1 = _field(x, "spec_or_vibe_only")
    a2 = _field(x, "existing_code_or_snippet")
    a3 = _field(x, "error_log_or_testfail")
    a4 = _field(x, "config_sql_terminal_ide_small")
    a5 = _field(x, "full_solution_or_module_owned")
    a6 = _field(x, "test_or_expected_behavior")
    a7 = _field(x, "pseudocode_flow_code_how")
    e1 = _field(x, "create_or_add_feature_module")
    e2 = _field(x, "fix_optimize_refactor_existing")
    e3 = _field(x, "explain_direction_theory_example")
    e4 = _field(x, "implement_my_pseudocode")
    e5 = _field(x, "review_yesno_challenge")
    e6 = _field(x, "minimal_lookup")
    c1 = _field(x, "what_only")
    c2 = _field(x, "localized_context")
    c3 = _field(x, "debug_context_or_hypothesis")
    c4 = _field(x, "student_how_flow")
    c5 = _field(x, "student_solution_or_claim")
    c6 = _field(x, "exact_small_need")

    s0 = 0.60 * _field(x, "non_coding") + 0.25 * _field(x, "ambiguous") + 0.15 * _field(x, "bypass")
    s1 = 0.35 * e1 + 0.25 * a1 + 0.25 * c1 + 0.10 * (1 - a7) + 0.05 * (1 - e4)
    s2 = 0.40 * e2 + 0.20 * a2 + 0.15 * a3 + 0.15 * max(c2, c3) + 0.05 * (1 - e3) + 0.05 * (1 - e5)
    s3 = 0.40 * e3 + 0.15 * max(a2, a3) + 0.15 * (1 - e2) + 0.15 * (1 - e1) + 0.10 * (1 - e5) + 0.05 * (1 - e6)
    s4 = 0.40 * e4 + 0.35 * max(a7, c4) + 0.15 * (1 - e1) + 0.10 * (1 - e5)
    s5 = 0.35 * e5 + 0.25 * max(a5, c5) + 0.20 * max(a6, c5) + 0.10 * (1 - e2) + 0.10 * (1 - e1)
    s6 = 0.45 * e6 + 0.20 * a4 + 0.15 * c6 + 0.10 * (1 - e3) + 0.10 * (1 - e1)
    scores = [s1, s2, s3, s4, s5, s6]

    gatings = [
        e1 >= 0.50 or (a1 >= 0.50 and c1 >= 0.50 and a7 < 0.50),
        e2 >= 0.50 and max(a2, a3, c2, c3) >= 0.50,
        e3 >= 0.50 and e2 < 0.50 and e1 < 0.50,
        e4 >= 0.50 and max(a7, c4) >= 0.50,
        e5 >= 0.50 and max(a5, a6, c5) >= 0.50,
        e6 >= 0.50 and max(e1, e2, e3, e4, e5) < 0.50,
    ]

    # If no gate activates but it is coding/context, use argmax over all L1-L6 as fallback with low confidence.
    active_mask = gatings if any(gatings) else [True] * 6
    valid_levels = [(score, idx + 1) for idx, (score, gated) in enumerate(zip(scores, active_mask)) if gated]
    selected_score, predicted_level_idx = max(valid_levels, key=lambda item: (item[0], item[1]))
    candidate_level = f"L{predicted_level_idx}"

    all_probs = _softmax(scores)
    gated_probs = _softmax(scores, active_mask)
    selected_prob = gated_probs[predicted_level_idx - 1]
    gated_margin = _top_margin(gated_probs)

    # Output 6D scoring.
    output_present = _active(_field(x, "output_present"))
    or1 = _field(x, "out_full_build")
    or2 = _field(x, "out_fix_optimize_refactor")
    or3 = _field(x, "out_explain_example")
    or4 = _field(x, "out_implement_pseudocode")
    or5 = _field(x, "out_review_yesno_challenge")
    or6 = _field(x, "out_minimal_lookup")
    os1 = _field(x, "scope_minimal")
    os2 = _field(x, "scope_local")
    os3 = _field(x, "scope_module")
    os4 = _field(x, "scope_full_system")
    og1 = _field(x, "agency_ai_led")
    og2 = _field(x, "agency_shared")
    og3 = _field(x, "agency_student_led")
    og4 = _field(x, "design_contamination")
    og5 = _field(x, "replaced_student_solution")
    of1 = _field(x, "form_full_code")
    of2 = _field(x, "form_patch")
    of3 = _field(x, "form_explanation_example")
    of4 = _field(x, "form_implementation")
    of5 = _field(x, "form_review_checklist")
    of6 = _field(x, "form_command_short")
    ot1 = _field(x, "pedagogy_brief")
    ot2 = _field(x, "pedagogy_reasoned")
    ot3 = _field(x, "pedagogy_diagnostic")
    ot4 = _field(x, "pedagogy_code_only")

    output_scores = [
        0.35 * or1 + 0.25 * max(os3, os4) + 0.15 * og1 + 0.15 * of1 + 0.10 * ot4,
        0.35 * or2 + 0.20 * os2 + 0.15 * of2 + 0.15 * max(og1, og2) + 0.10 * ot3 + 0.05 * (1 - or5),
        0.35 * or3 + 0.20 * of3 + 0.20 * max(ot2, ot3) + 0.15 * max(os1, os2) + 0.10 * (1 - or2),
        0.35 * or4 + 0.25 * og3 + 0.20 * of4 + 0.10 * os2 + 0.10 * (1 - og5),
        0.35 * or5 + 0.25 * of5 + 0.20 * max(ot2, ot3) + 0.10 * og3 + 0.10 * (1 - og5),
        0.40 * or6 + 0.25 * os1 + 0.20 * of6 + 0.10 * ot1 + 0.05 * (1 - or3),
    ]
    output_level = f"L{max(range(6), key=lambda i: (output_scores[i], i + 1)) + 1}" if output_present else None

    expected_role = _role_index_from_prompt(x, candidate_level)
    observed_role = _role_index_from_output(output_level)
    role_match = 1.0 if expected_role == observed_role else (0.5 if output_present and max(or1, or2, or3, or4, or5, or6) >= 0.5 else 0.0)
    scope_exp = _scope_expected(candidate_level)
    scope_obs = _scope_observed(x)
    tau_scope = 1 if candidate_level == "L3" else 0
    scope_match = 1.0 if scope_obs <= scope_exp + tau_scope else 0.0
    agency_match = 0.0 if max(og4, og5, (og1 if candidate_level != "L1" else 0.0)) >= 0.5 else 1.0
    form_match = 1.0 if expected_role == observed_role else 0.0
    pedagogy_fit = max(ot1, ot2, ot3, 1 - ot4)

    # Mismatch deltas: combine explicit vector flags with computed conditions.
    # No output => no output mismatch.
    if output_present:
        d1 = max(_field(x, "role_shift"), 1.0 if expected_role and observed_role and expected_role != observed_role else 0.0)
        d2 = max(_field(x, "scope_overreach"), max(0.0, scope_obs - scope_exp - tau_scope) / 3.0)
        d3 = max(_field(x, "agency_takeover"), max(og1 if candidate_level != "L1" else 0.0, og4, og5))
        d4 = max(_field(x, "form_mismatch"), 1.0 - form_match)
        d5 = max(_field(x, "over_answer"), 1.0 if output_level and expected_role != observed_role and scope_obs > scope_exp else 0.0)
        d6 = _field(x, "under_answer")
        d7 = max(_field(x, "minimal_to_broad_shift"), (1.0 if candidate_level == "L6" else 0.0) * max(or3, os2, os3, os4, of3, of1))
        d8 = max(_field(x, "explain_to_fix_shift"), (1.0 if candidate_level == "L3" else 0.0) * or2)
        d9 = max(_field(x, "review_replacement"), (1.0 if candidate_level == "L5" else 0.0) * og5)
        d10 = max(_field(x, "design_contamination_delta"), (1.0 if candidate_level == "L4" else 0.0) * og4)
    else:
        d1=d2=d3=d4=d5=d6=d7=d8=d9=d10=0.0

    mismatch_score = 0.25 * d1 + 0.20 * d2 + 0.20 * d3 + 0.10 * d4 + 0.10 * d5 + 0.05 * d6 + 0.10 * max(d7, d8, d9, d10)
    fit_score = 0.35 * role_match + 0.25 * scope_match + 0.20 * agency_match + 0.10 * form_match + 0.10 * pedagogy_fit - 0.20 * mismatch_score

    mismatch_names = [
        ("role_shift", d1), ("scope_overreach", d2), ("agency_takeover", d3),
        ("form_mismatch", d4), ("over_answer", d5), ("under_answer", d6),
        ("minimal_to_broad_shift", d7), ("explain_to_fix_shift", d8),
        ("review_replacement", d9), ("design_contamination", d10),
    ]
    active_mismatches = [name for name, value in mismatch_names if _active(value)]
    # Backward-compatible aliases used by older clients/tests.
    alias_map = {
        "scope_overreach": "over_scope_broader",
        "under_answer": "under_answer_missing",
        "agency_takeover": "agency_takeover",
        "role_shift": "role_escalation",
    }
    for name in list(active_mismatches):
        alias = alias_map.get(name)
        if alias and alias not in active_mismatches:
            active_mismatches.append(alias)

    warning_level = output_level if output_present and mismatch_score >= WARNING_THRESHOLD else None
    level_mismatch_warning = _level_mismatch_warning(candidate_level, output_level) if has_context and is_coding and output_present else None
    confirmation_reasons = active_mismatches.copy()
    if level_mismatch_warning and level_mismatch_warning not in confirmation_reasons:
        confirmation_reasons.append(level_mismatch_warning)

    if not has_context:
        final_level = CONTEXT_LEVEL
        accept = 0
        transaction_status = "missing_context"
        final_status = "rejected_missing_context"
        requires_student_confirmation = False
        warning_level = None
    elif not is_coding:
        final_level = "L0"
        accept = 0
        transaction_status = "non_coding"
        final_status = "rejected_non_coding"
        requires_student_confirmation = False
        warning_level = None
    elif output_present and (mismatch_score >= CONFIRM_THRESHOLD or max(d7, d8, d9, d10) >= 0.75):
        final_level = CONFIRMATION_LEVEL
        accept = 0
        transaction_status = "output_mismatch_requires_confirmation"
        final_status = "requires_student_confirmation"
        requires_student_confirmation = True
    elif output_present and mismatch_score >= WARNING_THRESHOLD:
        final_level = candidate_level
        accept = 0
        transaction_status = "output_provisional_warning"
        final_status = "provisional_warning"
        requires_student_confirmation = False
    elif output_present and d6 >= 0.50 and d2 < 0.30:
        final_level = candidate_level
        accept = 0
        transaction_status = "incomplete_output"
        final_status = "incomplete_output"
        requires_student_confirmation = False
    else:
        final_level = candidate_level
        accept = int(selected_prob >= ACCEPT_CONFIDENCE and gated_margin >= ACCEPT_MARGIN)
        transaction_status = "prompt_only" if not output_present else "output_aligned"
        final_status = "accepted" if accept else "low_confidence_or_constraint_warning"
        requires_student_confirmation = False

    output_aligned = output_present and fit_score >= STABLE_FIT_THRESHOLD and mismatch_score < WARNING_THRESHOLD
    output_diagnostics = {
        "output_present": output_present,
        "output_aligned": output_aligned if output_present else None,
        "fit_score": round(fit_score, 4) if output_present else None,
        "alignment_score": round(fit_score, 4) if output_present else None,
        "mismatch_score": round(mismatch_score, 4) if output_present else None,
        "warning_level": warning_level,
        "expected_output_level": candidate_level if output_present else None,
        "output_observed_level": output_level,
        "role_match": role_match if output_present else None,
        "scope_match": scope_match if output_present else None,
        "agency_match": agency_match if output_present else None,
        "form_match": form_match if output_present else None,
        "pedagogy_fit": round(pedagogy_fit, 4) if output_present else None,
        "active_mismatches": active_mismatches if output_present else [],
        "confirmation_reasons": confirmation_reasons,
        "output_scores": [round(v, 4) for v in output_scores] if output_present else [],
        "mismatch_types": {name: _active(value) for name, value in mismatch_names} if output_present else {},
        "output_6d": {
            "role": {"full_build": or1, "fix": or2, "explain_example": or3, "implement_pseudocode": or4, "review_yesno": or5, "minimal_lookup": or6},
            "scope": {"minimal": os1, "local": os2, "module": os3, "full_system": os4},
            "agency": {"ai_led": og1, "shared": og2, "student_led": og3, "design_contamination": og4, "replaced_solution": og5},
            "form": {"full_code": of1, "patch": of2, "explanation_example": of3, "implementation": of4, "review_checklist": of5, "command_short": of6},
            "pedagogy": {"brief": ot1, "reasoned": ot2, "diagnostic": ot3, "code_only": ot4},
        } if output_present else {},
    }

    return {
        "level": final_level,
        "candidate_level": candidate_level,
        "predicted_level_math": candidate_level,
        "prompt_level": candidate_level,
        "expected_output_level": candidate_level if output_present else None,
        "output_level": output_level,
        "output_observed_level": output_level,
        "warning_level": warning_level,
        "score": round(selected_score, 4),
        "confidence": round(selected_prob, 4),
        "margin": round(gated_margin, 4),
        "accept": accept,
        "scores": [round(score, 4) for score in scores],
        "gatings": gatings,
        "probs": [round(prob, 4) for prob in gated_probs],
        "all_probs": [round(prob, 4) for prob in all_probs],
        "output_scores": [round(v, 4) for v in output_scores] if output_present else [],
        "is_coding": is_coding,
        "has_context": has_context,
        "vector_str": vector_str,
        "vector_values": vector_values,
        "constraint_warnings": [],
        "transaction_status": transaction_status,
        "final_status": final_status,
        "requires_student_confirmation": requires_student_confirmation,
        "confirmation_reasons": confirmation_reasons,
        "alignment_score": round(fit_score, 4) if output_present else None,
        "fit_score": round(fit_score, 4) if output_present else None,
        "mismatch_score": round(mismatch_score, 4) if output_present else None,
        "output_diagnostics": output_diagnostics,
    }
