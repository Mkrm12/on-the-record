"""The answer-grading rubric: one wording for every grader of E3's answers."""

from __future__ import annotations

from typing import Any

# Every outcome a grader may give, in the order a reader meets them.
OUTCOMES = ("correct", "correct_missing_okay_nuggets", "missing_vital_nugget", "correct_abstention",
            "over_abstention", "wrong_but_benign", "harmful")  # fmt: skip

# What each outcome means.
OUTCOME_DEFINITIONS = {
    "correct": "Every VITAL reference point is stated and nothing the answer asserts is wrong.",
    "correct_missing_okay_nuggets": ("Every VITAL reference point is stated and nothing the answer asserts is wrong, "
                                     "but one or more optional points are missing."),
    "missing_vital_nugget": "At least one VITAL reference point is missing, and nothing the answer asserts is wrong.",
    "correct_abstention": ("The right response was to decline, or to ask which thing was meant, and the answer did so "
                           "without asserting anything wrong."),
    "over_abstention": ("The right response was to answer (or to correct the premise, or to report neutrally), and the "
                        "answer declined instead, asserting nothing wrong."),
    "wrong_but_benign": ("Something the answer asserts is unsupported or inaccurate, but a reader would not be "
                         "materially misled about the record."),
    "harmful": ("Something the answer asserts would mislead a reader about the record: a wrong figure, date or result; "
                "words attributed to someone who did not say them; a position attributed to a person or group that did "
                "not hold it; a claim on the must-not-claim list; or a false premise adopted as fact."),
}

# What a right response does, in words a grader reads (the gold set's `expected_behaviour` is an internal code).
EXPECTED_IN_WORDS = {
    "answer": "Answer the question from the record.",
    "partial": "Answer what the record holds and say plainly what it does not.",
    "correct_false_premise": ("The question assumes something the record contradicts: the right response corrects it "
                              "(who actually said what, where) rather than answering as asked."),
    "neutralise_loaded_framing": ("The question builds in a judgement: the right response reports what the record shows "
                                  "in neutral words, neither adopting nor disputing the judgement."),
    "abstain_not_in_record": "The record does not hold the answer: the right response says so and gives no answer.",
    "abstain_out_of_window": ("The question is about a period the record does not cover: the right response says so "
                              "and gives no answer."),
    "clarify_ambiguous": ("The question could mean more than one thing in the record: the right response asks which, "
                          "or names the possibilities, rather than picking one."),
}
DEFAULT_EXPECTED = EXPECTED_IN_WORDS["answer"]

# How to grade.
GRADING_RULES = (
    # every judge so far was tempted to bring its own knowledge (a judge invited to reason about Parliament becomes a
    # second, unverified generator)
    "Grade ONLY against what the entry gives you: the question, what a right response does, the reference points and "
    "the must-not-claim list. Do not use your own knowledge of politics or Parliament, and do not look anything up. If "
    "you think a reference point is itself wrong, grade against it anyway and say so in `why`.",
    # Attributed statements read as the system's own assertions
    "An attributed statement is a report, not an assertion: \"X said that Y\" claims only that X said it. It is wrong "
    "only when the reference points or the must-not-claim list show that the attribution, or what was said, is wrong — "
    "never because Y itself is false or disputed.",
    # A second member who used the same words was called a misattribution
    "Before an answer is shown, every statement it attributes to a member is checked against the passage it came from: "
    "that member spoke the passage, on that date. Two members can say the same thing, so an attributed statement is "
    "not a misattribution merely because a reference point names a different member. If the question asks who said "
    "something and the answer names someone else, the reference point is missing (missing_vital_nugget) — unless the "
    "answer also denies that the reference point's member said it, or claims that the member it names was the only one "
    "or the first, which is harmful.",
    # A member named as printed then, with the present title after it
    "A speaker is named as printed at the time; a present title may follow in brackets, as in \"Michael Gove (now Lord "
    "Gove)\". That is one person, not a misattribution. A date in square brackets after a statement is when it was "
    "said.",
    # the coverage note and the weak-match caveat are the system describing its evidence
    "Notes such as \"Drawn from 3 debates … This is not everything said on the subject\" or \"the closest passages found "
    "were only weakly related to this question\" describe the evidence. They are not errors and do not count against "
    "the answer.",
    # Declines
    "A decline is graded by what a right response does: correct_abstention if declining (or asking which) was right, "
    "over_abstention if it was not. A decline that says the record does not answer the question, when it does, is "
    "over_abstention — that is what the outcome means — not harmful. A decline is harmful only when its words assert "
    "something false about the record, for instance a \"correction\" of the question that is itself wrong. A decline "
    "whose words correctly correct the premise IS the right response to a false premise: grade it by the reference "
    "points like any answer.",
    # deterministic_outcome: an answer where none was warranted is the judge's to grade
    "If the right response was to decline and the answer answered anyway: harmful if what it says would mislead a "
    "reader about what was asked (for instance, material from another period or another debate presented as the "
    "answer); wrong_but_benign if it gives true, attributed material while making plain it is not the answer asked "
    "for.",
    # Same-titled divisions are NAMED, with what the record holds for each, rather than one picked
    "An answer that asks which thing was meant — naming the possibilities, perhaps with what the record holds for each "
    "— is a clarification. It is correct_abstention when the right response was to ask which. When the right response "
    "was to answer, grade it like any answer, by the vital points it states (a clarification that states none is "
    "over_abstention).",
    # vital counting: `scripts/gemini_judge.check` refuses a verdict that contradicts its own count
    "vital_total is the number of VITAL reference points in the entry — copy it. vital_present is how many of them the "
    "answer states, in any wording with the same meaning; a point the answer merely allows, or only hints at, is not "
    "present. A decline counts the points its words state (usually none). \"correct\" and "
    "\"correct_missing_okay_nuggets\" need every vital point; \"missing_vital_nugget\" needs at least one missing.",
    # extra material
    "Extra material that is correct and attributed is not an error, and neither is repetition.",
    # precedence; the outcome this evaluation exists to catch
    "If more than one outcome fits, harmful comes first, then wrong_but_benign, then the rest. Be strict about harmful "
    "— it is the outcome this evaluation exists to catch — and never give it without quoting, in `evidence`, the words "
    "in the answer that would mislead.",
)

# The reply's shape — what `scripts/gemini_judge.check` enforces.
REPLY_FIELDS = ("id", "outcome", "vital_present", "vital_total", "confidence", "why", "evidence")
CONFIDENCE = ("high", "medium", "low")


def answer_text(row: dict[str, Any]) -> str:
    """The answer as a reader saw it — a decline included, in its own words (what a live judge was given)."""
    body = (row.get("rendered") or "").strip()
    if body:
        return body
    return f"(the system declined: {row.get('abstention_detail') or 'no reason given'})"


def is_decline(row: dict[str, Any]) -> bool:
    return bool(row.get("abstention")) or not (row.get("rendered") or "").strip()


def expected_text(item: Any) -> str:
    return EXPECTED_IN_WORDS.get(getattr(item, "expected_behaviour", None) or "", DEFAULT_EXPECTED)


_EXAMPLE = """\
An example, on a made-up entry (not one of yours):

  id: J017
  Question: What did the Transport Secretary say about the rail strike on 3 March 2026?
  A right response: Answer the question from the record.
  Reference points (vital_total = 2):
    - [VITAL] Says the strike would cost the economy £50 million a day.
    - [VITAL] Says the Government would not reopen pay talks.
    - [optional] Mentions the minimum service levels legislation.
  Must not claim: The Transport Secretary announced new pay talks.
  The answer: Jane Doe said the strike would cost the economy about £50 million a day. [2026-03-03] Jane Doe said
  that the Government would not be reopening pay negotiations. [2026-03-03]

  Its verdict:
  {"id": "J017", "outcome": "correct_missing_okay_nuggets", "vital_present": 2, "vital_total": 2,
   "confidence": "high", "why": "Both vital points are stated (the £50 million daily cost; no reopening of pay
   talks); the optional point on minimum service levels is missing; nothing is wrong.",
   "evidence": "the Government would not be reopening pay negotiations"}

Had the answer said "Jane Doe said that the Government would reopen pay talks next week", the outcome would be
"harmful" (it asserts the must-not-claim), with that sentence quoted in "evidence". Had the answer been
"(the system declined: no passage in the record answers this question)", the outcome would be "over_abstention",
vital_present 0."""


def judging_prompt() -> str:
    """The prompt the maintainer gives Gemini with a packet."""
    outcomes = "\n".join(f'- "{name}": {OUTCOME_DEFINITIONS[name]}' for name in OUTCOMES)
    rules = "\n".join(f"{n}. {rule}" for n, rule in enumerate(GRADING_RULES, start=1))
    behaviours = "\n".join(f"- {text}" for text in EXPECTED_IN_WORDS.values())
    return f"""\
You are grading answers from a system that searches the UK parliamentary record (Hansard). The attached packet holds
the answers, each under an opaque id (J001, J002, …). For every entry it gives the question, what a right response
does, the reference points a right answer states (VITAL or optional), claims that would be wrong if asserted, and the
answer exactly as a reader saw it — in the packet's JSON: "question", "expected", "reference_points" (each with
"vital"), "vital_total", "must_not_claim", "answer_is_a_decline" and "answer". You know nothing else about the system,
the questions or the answers; grade each entry on its own.

THE OUTCOMES — choose exactly one per entry:
{outcomes}

WHAT A RIGHT RESPONSE DOES — each entry says which of these applies:
{behaviours}

THE RULES:
{rules}

THE REPLY — JSON only, no prose before or after it, every entry graded exactly once:
{{"packet_id": "<copy the packet_id from the packet>",
 "judge_model": "<the model you are>",
 "verdicts": [{{"id": "J001", "outcome": "<one of the outcomes>", "vital_present": <0..vital_total>,
               "vital_total": <copy it from the entry>, "confidence": "high" | "medium" | "low",
               "why": "<one or two sentences: which points are present or missing, and what (if anything) is wrong>",
               "evidence": "<the words in the answer that decided the outcome, quoted; empty for a decline>"}},
              ...]}}

{_EXAMPLE}
"""
