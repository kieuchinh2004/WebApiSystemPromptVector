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
        name="Out of coding scope / insufficient transaction",
        core_definition="Prompt is non-coding, too ambiguous, or contains bypass/self-label behavior.",
        artifact="No reliable programming-learning artifact.",
        expectation="No clear coding expectation or unsafe self-label/bypass.",
        contribution="Student contribution cannot be assessed under the coding rubric.",
        output_should="Ask for clarification or reject instead of forcing L1-L6.",
        output_mismatch_risk="Forcing a coding level when evidence is insufficient.",
    ),
    "L1": RubricLevel(
        code="L1",
        name="AI builds from scratch / feature-module / vibe code",
        core_definition="Student describes WHAT in natural/business language or asks AI to create/add a substantial product, feature, module, app, or solution.",
        artifact="Usually only spec/user story/vibe code; no function-level pseudocode/HOW.",
        expectation="AI writes/builds/generates from scratch or adds a substantial feature/module.",
        contribution="Student contributes requirements/business logic, not implementation flow.",
        output_should="A full feature/module/solution is aligned if exactly requested.",
        output_mismatch_risk="Natural-language business description may be mistaken for L4, but without code-oriented HOW it remains L1.",
    ),
    "L2": RubricLevel(
        code="L2",
        name="Fix / optimize / refactor existing code",
        core_definition="Student has an existing artifact/context and asks AI to directly fix, debug, optimize, or refactor it.",
        artifact="Existing code, snippet, error, log, failed test, or local context.",
        expectation="AI directly edits/patches/improves the artifact.",
        contribution="Student localizes the problem but delegates the change to AI.",
        output_should="Focused fix/patch/refactor within the supplied artifact.",
        output_mismatch_risk="AI expands to a full module/project beyond a local fix.",
    ),
    "L3": RubricLevel(
        code="L3",
        name="Explanation / direction / theory with example",
        core_definition="Student asks for direction, explanation, theory with examples, mechanism, or code explanation.",
        artifact="May include code/error snippet, but the goal is understanding rather than direct fixing.",
        expectation="AI explains, teaches, gives direction, or provides examples.",
        contribution="Student identifies a learning question or wants to understand an approach/mechanism.",
        output_should="Explanation, reasoning, direction, and optionally a small illustrative example.",
        output_mismatch_risk="AI fixes code directly instead of explaining.",
    ),
    "L4": RubricLevel(
        code="L4",
        name="Implement my pseudocode / code-oriented flow",
        core_definition="Student gives pseudocode, function-level flow, algorithm, or explicit HOW; AI implements that design.",
        artifact="Pseudo code, flow code, function steps, algorithm, transformation, or technical HOW.",
        expectation="AI implements according to the student's provided HOW.",
        contribution="Student owns the design/flow/algorithm.",
        output_should="Implementation faithful to the supplied pseudocode/flow.",
        output_mismatch_risk="AI adds/changes architecture or design not requested; vibe-code/spec-only wrongly treated as L4.",
    ),
    "L5": RubricLevel(
        code="L5",
        name="Yes/no / review / challenge / verification",
        core_definition="Student has a claim, answer, solution, or decision and asks AI to verify, review, challenge, or answer yes/no with reasons.",
        artifact="Student-owned solution, claim, test, edge case, or answer to check.",
        expectation="AI acts as reviewer/challenger/verifier, not primary author.",
        contribution="Student has produced something or taken a position before asking AI.",
        output_should="Yes/no with reasoning, review comments, edge cases, counterarguments, or tests.",
        output_mismatch_risk="AI rewrites/replaces the solution instead of reviewing/challenging it.",
    ),
    "L6": RubricLevel(
        code="L6",
        name="Minimal lookup / small technical item",
        core_definition="Student asks one narrow item: short theory definition, syntax/API, config, SQL, terminal, or IDE setting; no mechanism explanation or extended example.",
        artifact="Usually no task artifact, or only a very small environmental/config/query item.",
        expectation="AI returns a brief exact answer.",
        contribution="Student knows precisely the missing micro-piece.",
        output_should="Short definition/command/syntax/config answer only.",
        output_mismatch_risk="AI expands into L3 explanation/examples or L1/L2 code work.",
    ),
}

PROMPT_DIMENSIONS = {
    "Artifact": "What material does the student provide: spec/vibe, existing code, error/log, config/SQL/terminal/IDE, full solution, tests, or pseudocode/HOW?",
    "Expectation": "What does the student expect AI to do: create/add, fix/optimize, explain/example, implement pseudocode, review/challenge, or minimal lookup?",
    "Contribution": "What has the student contributed: WHAT-only, localized context, debug hypothesis, code-oriented flow, own solution/claim, or exact small need?",
}

OUTPUT_DIMENSIONS = {
    "Role": "What did AI actually do: full build, fix, explain/example, implement pseudocode, review/yes-no, or minimal lookup?",
    "Scope": "How broad is the output: minimal, local, module, or full-system?",
    "Agency": "Who controls decisions: AI-led, shared, student-led, design contamination, or replacement of student solution?",
    "Form": "What form did AI answer with: full code, patch, explanation+example, implementation, review/checklist, or short command?",
    "Pedagogy": "How much learning support is present: brief, reasoned, diagnostic, or code-only/copy-paste?",
    "Mismatch": "How does output deviate from expectation: role shift, scope overreach, agency takeover, form mismatch, over/under-answer, or level-specific shifts?",
}


def rubric_as_dict() -> dict[str, Any]:
    return {
        "levels": {key: asdict(value) for key, value in RUBRIC_LEVELS.items()},
        "prompt_dimensions": PROMPT_DIMENSIONS,
        "output_dimensions": OUTPUT_DIMENSIONS,
    }
