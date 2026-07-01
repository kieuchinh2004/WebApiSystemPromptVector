"""Deterministic APC v4 transaction scoring engine.

Rubric-grounded vector model:
- Rubric is the theoretical standard for L0-L6.
- Prompt vector operationalizes Artifact / Expectation / Contribution to select
  a candidate student level.
- Output vector does NOT re-score the student's capability.  It only checks
  whether AI_OUTPUT matched the student's expectation.  If it did not match, the
  engine keeps candidate_level but returns a confirmation-required status.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from rubric_schema import RUBRIC_LEVELS


PROMPT_VECTOR_SIZE = 26
VECTOR_SIZE = 44
TAU = 8.0
ACCEPT_CONFIDENCE = 0.60
ACCEPT_MARGIN = 0.15
GATING_THRESHOLD = 0.50
ALIGNMENT_ACCEPT_THRESHOLD = 0.75
MISMATCH_REJECT_THRESHOLD = 0.30

LEVELS = tuple(f"L{i}" for i in range(1, 7))
CONTEXT_LEVEL = "Thieu context"
CONFIRMATION_LEVEL = "Can hoi lai SV"


def _parse_vector(vector_input: str | Sequence[float | int]) -> list[float]:
    """Parse either legacy 26D prompt vector or new 44D transaction vector."""

    if isinstance(vector_input, str):
        vector_str = "".join(vector_input.split())
        if len(vector_str) not in (PROMPT_VECTOR_SIZE, VECTOR_SIZE) or not all(c in "01" for c in vector_str):
            raise ValueError(
                f"Vector string must contain exactly {PROMPT_VECTOR_SIZE} or {VECTOR_SIZE} binary characters, "
                f"got {len(vector_str)}: {vector_str!r}"
            )
        values = [float(c) for c in vector_str]
    else:
        if len(vector_input) not in (PROMPT_VECTOR_SIZE, VECTOR_SIZE):
            raise ValueError(
                f"Vector list must contain exactly {PROMPT_VECTOR_SIZE} or {VECTOR_SIZE} elements, "
                f"got {len(vector_input)}"
            )
        values = [float(v) for v in vector_input]
        invalid = [v for v in values if not 0.0 <= v <= 1.0]
        if invalid:
            raise ValueError(f"Vector values must be in [0, 1], got invalid values: {invalid[:5]}")

    if len(values) == PROMPT_VECTOR_SIZE:
        values = values + [0.0] * (VECTOR_SIZE - PROMPT_VECTOR_SIZE)
    return values


def _binary_view(values: Sequence[float]) -> str:
    return "".join("1" if value >= GATING_THRESHOLD else "0" for value in values)


def _softmax(scores: Sequence[float], active: Sequence[bool] | None = None) -> list[float]:
    if active is None:
        active = [True] * len(scores)
    if len(scores) != len(active):
        raise ValueError("scores and active mask must have the same length")

    active_scores = [score for score, enabled in zip(scores, active) if enabled]
    if not active_scores:
        raise ValueError("At least one active level is required")

    max_logit = max(TAU * score for score in active_scores)
    exps: list[float] = []
    for score, enabled in zip(scores, active):
        exps.append(math.exp(TAU * score - max_logit) if enabled else 0.0)
    total = sum(exps)
    return [value / total for value in exps]


def _top_margin(probs: Sequence[float]) -> float:
    sorted_probs = sorted(probs, reverse=True)
    if len(sorted_probs) < 2:
        return sorted_probs[0] if sorted_probs else 0.0
    return sorted_probs[0] - sorted_probs[1]


def _active(value: float) -> bool:
    return value >= GATING_THRESHOLD


def _alignment_score(role: float, scope: float, agency: float, form: float, pedagogy: float) -> float:
    # Role/scope/agency are weighted highest because they protect against output-driven misclassification.
    return 0.25 * role + 0.25 * scope + 0.20 * agency + 0.15 * form + 0.15 * pedagogy


def _mismatch_score(over_scope: float, under_answer: float, role_escalation: float, agency_takeover: float, form_pedagogy: float) -> float:
    # Over-scope and agency takeover are most harmful in learning analytics because AI behavior can hide student agency.
    return 0.25 * over_scope + 0.15 * under_answer + 0.20 * role_escalation + 0.25 * agency_takeover + 0.15 * form_pedagogy


def _infer_output_level(
    complete_solution: float,
    direct_code_patch: float,
    review_feedback: float,
    tests: float,
    explanation: float,
    narrow_reference: float,
) -> str | None:
    """Classify which rubric level the AI_OUTPUT itself resembles, based on its form.

    Ordered from most to least "delegated" so that when several forms are present
    at once (e.g. an explanation padded with a full worked code example), the
    reading reflects the broadest scope the output actually delivered.
    """

    if _active(complete_solution):
        return "L1"
    if _active(direct_code_patch):
        return "L2"
    if _active(review_feedback) or _active(tests):
        return "L5"
    if _active(explanation):
        return "L3"
    if _active(narrow_reference):
        return "L6"
    return None


def _confirmation_reasons(
    names_and_values: Sequence[tuple[str, float]],
    alignment_score: float,
    mismatch_score: float,
    candidate_level: str,
    output_level: str | None,
) -> list[str]:
    reasons = [name for name, value in names_and_values if _active(value)]
    if alignment_score < ALIGNMENT_ACCEPT_THRESHOLD:
        reasons.append(f"alignment_score_below_{ALIGNMENT_ACCEPT_THRESHOLD:.2f}")
    if mismatch_score >= MISMATCH_REJECT_THRESHOLD:
        reasons.append(f"mismatch_score_at_or_above_{MISMATCH_REJECT_THRESHOLD:.2f}")
    # Warn using the level the OUTPUT itself resembles (not the prompt's
    # candidate_level), so SV sees e.g. "AI answered like L3" instead of a
    # generic mismatch code when the mismatch is really a level shift.
    if output_level is not None and output_level != candidate_level:
        output_rubric = RUBRIC_LEVELS[output_level]
        candidate_rubric = RUBRIC_LEVELS[candidate_level]
        reasons.append(
            f"output_behaves_like_{output_level}_not_{candidate_level}: "
            f"AI_OUTPUT matches '{output_rubric.name}' pattern ({output_rubric.core_definition}) "
            f"but candidate level is {candidate_level} ('{candidate_rubric.name}'), which expects: "
            f"{candidate_rubric.output_should}"
        )
    return reasons


def compute_apc_v4(vector_input: str | Sequence[float | int]) -> dict[str, Any]:
    """Compute APC scores, gated prediction, confidence, and transaction status."""

    x = _parse_vector(vector_input)
    vector_values = [round(value, 4) for value in x]
    vector_str = _binary_view(x)

    is_coding = _active(x[0])
    has_context = _active(x[1])

    a1, a2, a3, a4, a5, a6, a7 = x[2:9]
    d1, d2, d3, d4, d5, d6, d7 = x[9:16]
    r1, r2, r3, r4, r5, r6, r7, r8, r9, r10 = x[16:26]

    (
        o_present,
        o_role_aligned,
        o_scope_aligned,
        o_agency_aligned,
        o_form_aligned,
        o_pedagogy_aligned,
        o_asks_clarification,
        o_complete_solution,
        o_direct_code_patch,
        o_explanation,
        o_review_feedback,
        o_tests,
        o_narrow_reference,
        m_over_scope,
        m_under_answer,
        m_role_escalation,
        m_agency_takeover,
        m_form_pedagogy_mismatch,
    ) = x[26:44]

    constraint_warnings: list[str] = []
    if a6 >= GATING_THRESHOLD and a7 >= GATING_THRESHOLD:
        constraint_warnings.append(
            "Invalid feature combination: a6(extensive_context) and a7(small_snippet) are both active. "
            "Scoring kept the values but accept is forced to 0."
        )

    # Prompt-side score formulas.  These select candidate_level only.
    s1 = (
        0.30 * r1
        + 0.10 * (1 - a1)
        + 0.05 * (1 - a2)
        + 0.05 * (1 - a3)
        + 0.05 * (1 - a4)
        + 0.05 * (1 - a5)
        + 0.15 * (1 - d1)
        + 0.10 * (1 - d2)
        + 0.15 * (1 - r7)
    )
    s2 = (
        0.10 * r2
        + 0.10 * r3
        + 0.10 * r4
        + 0.10 * r5
        + 0.20 * a1
        + 0.10 * a5
        + 0.10 * d5
        + 0.05 * d6
        + 0.10 * (1 - r6)
        + 0.05 * (1 - r8)
    )
    s3 = (
        0.40 * r6
        + 0.10 * a2
        + 0.10 * a7
        + 0.15 * (1 - a6)
        + 0.10 * (1 - r8)
        + 0.05 * (1 - r2)
        + 0.05 * (1 - r10)
        + 0.05 * (1 - d5)
    )
    s4 = (
        0.25 * d1
        + 0.15 * d2
        + 0.10 * d3
        + 0.10 * d4
        + 0.25 * r7
        + 0.10 * (1 - r1)
        + 0.05 * (1 - r8)
    )
    s5 = (
        0.25 * r8
        + 0.20 * a6
        + 0.10 * r9
        + 0.10 * d7
        + 0.05 * a2
        + 0.10 * (1 - r2)
        + 0.10 * (1 - a7)
        + 0.10 * (1 - r1)
    )
    s6 = (
        0.40 * r10
        + 0.15 * a7
        + 0.15 * (1 - r1)
        + 0.10 * (1 - d1)
        + 0.10 * (1 - r6)
        + 0.10 * (1 - a6)
    )
    scores = [s1, s2, s3, s4, s5, s6]

    gatings = [
        True,
        True,
        True,
        (d1 >= GATING_THRESHOLD or d2 >= GATING_THRESHOLD or d3 >= GATING_THRESHOLD or d4 >= GATING_THRESHOLD)
        and r7 >= GATING_THRESHOLD,
        a6 >= GATING_THRESHOLD and r8 >= GATING_THRESHOLD,
        r10 >= GATING_THRESHOLD,
    ]

    valid_levels: list[tuple[float, int]] = []
    for idx, (score, gated) in enumerate(zip(scores, gatings)):
        if gated:
            valid_levels.append((score, idx + 1))

    selected_score, predicted_level_idx = max(valid_levels, key=lambda item: (item[0], item[1]))
    candidate_level = f"L{predicted_level_idx}"

    all_probs = _softmax(scores)
    gated_probs = _softmax(scores, gatings)
    selected_prob = gated_probs[predicted_level_idx - 1]
    gated_margin = _top_margin(gated_probs)

    output_present = _active(o_present)
    output_level = (
        _infer_output_level(o_complete_solution, o_direct_code_patch, o_review_feedback, o_tests, o_explanation, o_narrow_reference)
        if output_present
        else None
    )

    # For levels where extensive explanation is not required, pedagogy is considered OK unless an explicit mismatch fires.
    pedagogy_effective = o_pedagogy_aligned
    if output_present and candidate_level in {"L1", "L2", "L6"} and not _active(m_form_pedagogy_mismatch):
        pedagogy_effective = max(pedagogy_effective, 1.0)

    alignment_score = _alignment_score(
        o_role_aligned,
        o_scope_aligned,
        o_agency_aligned,
        o_form_aligned,
        pedagogy_effective,
    )
    mismatch_score = _mismatch_score(
        m_over_scope,
        m_under_answer,
        m_role_escalation,
        m_agency_takeover,
        m_form_pedagogy_mismatch,
    )

    mismatch_names = [
        ("over_scope_broader", m_over_scope),
        ("under_answer_missing", m_under_answer),
        ("role_escalation", m_role_escalation),
        ("agency_takeover", m_agency_takeover),
        ("form_or_pedagogy_mismatch", m_form_pedagogy_mismatch),
    ]
    active_mismatches = [name for name, value in mismatch_names if _active(value)]
    output_aligned = (
        output_present
        and alignment_score >= ALIGNMENT_ACCEPT_THRESHOLD
        and mismatch_score < MISMATCH_REJECT_THRESHOLD
        and not active_mismatches
    )

    if not has_context:
        final_level = CONTEXT_LEVEL
        accept = 0
        transaction_status = "missing_context"
        final_status = "rejected_missing_context"
        requires_student_confirmation = False
        confirmation_reasons: list[str] = []
    elif not is_coding:
        final_level = "L0"
        accept = 0
        transaction_status = "non_coding"
        final_status = "rejected_non_coding"
        requires_student_confirmation = False
        confirmation_reasons = []
    elif output_present and not output_aligned:
        final_level = CONFIRMATION_LEVEL
        accept = 0
        transaction_status = "output_mismatch_requires_confirmation"
        final_status = "requires_student_confirmation"
        requires_student_confirmation = True
        confirmation_reasons = _confirmation_reasons(mismatch_names, alignment_score, mismatch_score, candidate_level, output_level)
    else:
        final_level = candidate_level
        accept = int(
            selected_prob >= ACCEPT_CONFIDENCE
            and gated_margin >= ACCEPT_MARGIN
            and not constraint_warnings
        )
        transaction_status = "prompt_only" if not output_present else "output_aligned"
        final_status = "accepted" if accept else "low_confidence_or_constraint_warning"
        requires_student_confirmation = False
        confirmation_reasons = []

    output_diagnostics = {
        "output_present": output_present,
        "output_aligned": output_aligned if output_present else None,
        "alignment_score": round(alignment_score, 4) if output_present else None,
        "mismatch_score": round(mismatch_score, 4) if output_present else None,
        "alignment_threshold": ALIGNMENT_ACCEPT_THRESHOLD,
        "mismatch_threshold": MISMATCH_REJECT_THRESHOLD,
        "role_aligned": _active(o_role_aligned) if output_present else None,
        "scope_aligned": _active(o_scope_aligned) if output_present else None,
        "agency_aligned": _active(o_agency_aligned) if output_present else None,
        "form_aligned": _active(o_form_aligned) if output_present else None,
        "pedagogy_aligned": _active(pedagogy_effective) if output_present else None,
        "asks_clarification": _active(o_asks_clarification) if output_present else None,
        "has_mismatch": bool(active_mismatches) if output_present else None,
        "active_mismatches": active_mismatches if output_present else [],
        "confirmation_reasons": confirmation_reasons,
        "output_level": output_level,
        "output_level_name": RUBRIC_LEVELS[output_level].name if output_level else None,
        "mismatch_types": {
            "over_scope_broader": _active(m_over_scope),
            "under_answer_missing": _active(m_under_answer),
            "role_escalation": _active(m_role_escalation),
            "agency_takeover": _active(m_agency_takeover),
            "form_or_pedagogy_mismatch": _active(m_form_pedagogy_mismatch),
        } if output_present else {},
        "output_forms": {
            "complete_solution": _active(o_complete_solution),
            "direct_code_patch": _active(o_direct_code_patch),
            "explanation": _active(o_explanation),
            "review_feedback": _active(o_review_feedback),
            "tests": _active(o_tests),
            "narrow_reference": _active(o_narrow_reference),
        } if output_present else {},
    }

    return {
        "level": final_level,
        "candidate_level": candidate_level,
        "predicted_level_math": candidate_level,  # backward compatibility
        "score": round(selected_score, 4),
        "confidence": round(selected_prob, 4),
        "margin": round(gated_margin, 4),
        "accept": accept,
        "scores": [round(score, 4) for score in scores],
        "gatings": gatings,
        "probs": [round(prob, 4) for prob in gated_probs],
        "all_probs": [round(prob, 4) for prob in all_probs],
        "is_coding": is_coding,
        "has_context": has_context,
        "vector_str": vector_str,
        "vector_values": vector_values,
        "constraint_warnings": constraint_warnings,
        "transaction_status": transaction_status,
        "final_status": final_status,
        "requires_student_confirmation": requires_student_confirmation,
        "confirmation_reasons": confirmation_reasons,
        "alignment_score": round(alignment_score, 4) if output_present else None,
        "mismatch_score": round(mismatch_score, 4) if output_present else None,
        "output_diagnostics": output_diagnostics,
    }
