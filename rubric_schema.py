"""Rubric-grounded APC definitions.

This module is intentionally data-only.  It makes the theoretical rubric visible
outside the prompt/scoring formula so reviewers can inspect how the vector engine
is grounded in L0-L6 definitions and the three prompt dimensions:
Artifact, Expectation, and Contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class RubricLevel:
    code: str
    name: str
    core_definition: str
    artifact: str
    expectation: str
    contribution: str
    output_should: str
    output_mismatch_risk: str


RUBRIC_LEVELS: dict[str, RubricLevel] = {
    "L0": RubricLevel(
        code="L0",
        name="Out of coding scope / insufficient academic transaction",
        core_definition="Prompt is not about programming/technical work, or cannot be evaluated as a coding-learning interaction.",
        artifact="No relevant programming artifact is required.",
        expectation="Not a coding-related expectation.",
        contribution="Student contribution cannot be assessed under APC coding rubric.",
        output_should="Reject as non-coding or missing-context instead of forcing L1-L6.",
        output_mismatch_risk="Forcing a coding APC level for non-coding material.",
    ),
    "L1": RubricLevel(
        code="L1",
        name="Full Delegation",
        core_definition="Student delegates creation of a complete solution to AI with little/no supplied artifact or design.",
        artifact="Usually no code/error/log artifact, or only problem statement/specification.",
        expectation="AI writes/builds/generates the solution from scratch.",
        contribution="Student contributes requirements, not solution design or verification evidence.",
        output_should="May provide a complete solution if that was exactly requested, but should avoid pretending this reflects high student agency.",
        output_mismatch_risk="AI gives advanced review/explanation and system overestimates student contribution.",
    ),
    "L2": RubricLevel(
        code="L2",
        name="Localized Delegation",
        core_definition="Student provides a local artifact/context but asks AI to modify, fix, refactor, convert, or add directly.",
        artifact="Code snippet, error trace, log, config, schema, or specific bug location may be present.",
        expectation="AI directly fixes/edits/patches/converts the artifact.",
        contribution="Student has localized the work, possibly with a location or hypothesis, but still delegates the change.",
        output_should="Return a focused patch/fix and concise rationale within the supplied local scope.",
        output_mismatch_risk="AI expands into whole-project architecture or full rewrite beyond local request.",
    ),
    "L3": RubricLevel(
        code="L3",
        name="Exploratory Understanding",
        core_definition="Student primarily asks to understand why/how code, an error, or a concept works, rather than asking AI to fix it.",
        artifact="May contain code/error snippet as evidence for explanation; often a small artifact.",
        expectation="AI explains mechanism, cause, concept, or reasoning.",
        contribution="Student identifies a learning question and seeks conceptual understanding.",
        output_should="Explain causes/mechanisms clearly; code is optional and should not replace explanation.",
        output_mismatch_risk="AI silently gives a fixed solution, causing agency takeover and form mismatch.",
    ),
    "L4": RubricLevel(
        code="L4",
        name="Guided Construction",
        core_definition="Student supplies their own design/algorithm/formula/flow and asks AI to implement that design.",
        artifact="May or may not include code, but includes student-owned design evidence.",
        expectation="AI implements according to the student's design, not inventing the main design itself.",
        contribution="Student contributes substantive design: algorithm, computation logic, structure, or data flow.",
        output_should="Implement faithfully following the provided design and flag deviations/assumptions.",
        output_mismatch_risk="AI replaces student design with its own solution or rewrites the task goal.",
    ),
    "L5": RubricLevel(
        code="L5",
        name="Reflective Verification",
        core_definition="Student has a complete or near-complete solution and asks AI to review, verify, test, or find edge cases.",
        artifact="Extensive/complete solution or enough context for evaluation, often with tests/expected I/O.",
        expectation="AI acts as reviewer/validator, not as primary author.",
        contribution="Student owns the solution and uses AI for reflection and quality control.",
        output_should="Provide review findings, risks, edge cases, tests, and reasoning; avoid unnecessary full rewrite.",
        output_mismatch_risk="AI rewrites everything or gives only superficial approval without evidence.",
    ),
    "L6": RubricLevel(
        code="L6",
        name="Minimal Augmentation",
        core_definition="Student asks for a narrow technical reference: syntax, API, command, parameter, regex, or small lookup.",
        artifact="Usually none or a tiny snippet; context is narrow and bounded.",
        expectation="AI returns a short, precise reference or command.",
        contribution="Student retains almost all problem-solving agency; AI supplies one missing micro-piece.",
        output_should="Answer narrowly and briefly; avoid generating the surrounding solution.",
        output_mismatch_risk="AI over-scopes into full implementation, falsely making transaction look like delegation.",
    ),
}


PROMPT_DIMENSIONS = {
    "Artifact": "What evidence/material does the student provide: code, errors, logs, schema, config, whole solution, or tiny snippet?",
    "Expectation": "What action does the student expect from AI: create, fix, explain, implement design, review, test, or lookup?",
    "Contribution": "What intellectual work has the student already contributed: design, algorithm, computation logic, structure, flow, hypothesis, location, or tests?",
}


OUTPUT_DIMENSIONS = {
    "RoleAlignment": "Does AI keep the role requested by the student (explainer, fixer, reviewer, implementer, lookup assistant)?",
    "ScopeAlignment": "Does AI stay within the requested scope rather than under-answering or over-solving?",
    "AgencyAlignment": "Does AI preserve the student's learning agency and contribution?",
    "FormAlignment": "Does AI use the requested/appropriate form: code patch, explanation, review, tests, or narrow reference?",
    "PedagogyAlignment": "Does AI provide the right amount of reasoning/transparency for the candidate level?",
    "MismatchType": "If alignment fails, which mismatch occurred: over-scope, under-answer, role escalation, agency takeover, or form/pedagogy mismatch?",
}


def rubric_as_dict() -> dict[str, Any]:
    return {
        "levels": {key: asdict(value) for key, value in RUBRIC_LEVELS.items()},
        "prompt_dimensions": PROMPT_DIMENSIONS,
        "output_dimensions": OUTPUT_DIMENSIONS,
    }


