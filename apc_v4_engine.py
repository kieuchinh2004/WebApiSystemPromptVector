"""Deterministic APC v4 scoring engine.

The methodology defines phi(p) in [0, 1]^24.  The deployed API prepends two
operational bits:

    index 0: is_coding
    index 1: has_context

so this module accepts a 26-dimensional vector in [0, 1].  Binary strings are
still accepted as a compatibility shorthand.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


VECTOR_SIZE = 26
TAU = 8.0
ACCEPT_CONFIDENCE = 0.60
ACCEPT_MARGIN = 0.15
GATING_THRESHOLD = 0.50

LEVELS = tuple(f"L{i}" for i in range(1, 7))
CONTEXT_LEVEL = "Thieu context"


def _parse_vector(vector_input: str | Sequence[float | int]) -> list[float]:
    if isinstance(vector_input, str):
        vector_str = "".join(vector_input.split())
        if len(vector_str) != VECTOR_SIZE or not all(c in "01" for c in vector_str):
            raise ValueError(
                f"Vector string must contain exactly {VECTOR_SIZE} binary characters, "
                f"got {len(vector_str)}: {vector_str!r}"
            )
        return [float(c) for c in vector_str]

    if len(vector_input) != VECTOR_SIZE:
        raise ValueError(
            f"Vector list must contain exactly {VECTOR_SIZE} elements, got {len(vector_input)}"
        )

    values = [float(v) for v in vector_input]
    invalid = [v for v in values if not 0.0 <= v <= 1.0]
    if invalid:
        raise ValueError(f"All vector values must be in [0, 1], got invalid values: {invalid[:5]}")
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

    # Numerical stability: subtract max active logit.
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


def compute_apc_v4(vector_input: str | Sequence[float | int]) -> dict[str, Any]:
    """Compute APC scores, gated prediction, confidence, and accept status."""

    x = _parse_vector(vector_input)
    vector_values = [round(value, 4) for value in x]
    vector_str = _binary_view(x)

    is_coding = x[0] >= GATING_THRESHOLD
    has_context = x[1] >= GATING_THRESHOLD

    a1, a2, a3, a4, a5, a6, a7 = x[2:9]
    d1, d2, d3, d4, d5, d6, d7 = x[9:16]
    r1, r2, r3, r4, r5, r6, r7, r8, r9, r10 = x[16:26]

    constraint_warnings: list[str] = []
    if a6 >= GATING_THRESHOLD and a7 >= GATING_THRESHOLD:
        constraint_warnings.append(
            "Invalid feature combination: a6(extensive_context) and a7(small_snippet) "
            "are both active. Scoring kept the values but accept is forced to 0."
        )

    # Score formulas from Methodology_APC_24F_v4_Corrected.docx Appendix A.
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
    predicted_level_math = f"L{predicted_level_idx}"

    # Keep both probability views.  all_probs is useful for diagnostics; gated_probs
    # is the probability distribution used for confidence/accept.
    all_probs = _softmax(scores)
    gated_probs = _softmax(scores, gatings)
    selected_prob = gated_probs[predicted_level_idx - 1]
    gated_margin = _top_margin(gated_probs)

    if not has_context:
        final_level = CONTEXT_LEVEL
        accept = 0
    elif not is_coding:
        final_level = "L0"
        accept = 0
    else:
        final_level = predicted_level_math
        accept = int(
            selected_prob >= ACCEPT_CONFIDENCE
            and gated_margin >= ACCEPT_MARGIN
            and not constraint_warnings
        )

    return {
        "level": final_level,
        "predicted_level_math": predicted_level_math,
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
    }
