"""The gold lint L0-L5: a reference point must test what the question asks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from goldset.lint import CODES, Finding, lint, lint_all
from goldset.schema import Evidence, GoldItem, Nugget, from_dict

ROOT = Path(__file__).resolve().parents[1]
FROZEN = json.loads((ROOT / "tests" / "data" / "gold_lint_offenders_2026-10-09.json").read_text(encoding="utf-8"))


def item(question, nuggets, breadth="broad", min_points=1, **kw) -> GoldItem:
    base = dict(id="hum9001", question=question, category="paraphrase", criticality="major",
                expected_behaviour="answer", provenance="passage_llm", breadth=breadth,
                min_points=min_points if breadth == "broad" else None,
                evidence=(Evidence("SPEECH-1", "A long enough quotation from the speech itself."),),
                nuggets=tuple(nuggets))
    return GoldItem(**(base | kw))


def codes(i: GoldItem) -> set[str]:
    return {f.code for f in lint(i)}


V = True


# --- each rule, both ways -------------------------------------------------------------------------------------------


def test_the_codes_are_the_six_rules():
    assert CODES == ("L0", "L1", "L2", "L3", "L4", "L5")


def test_L0_an_unclassified_item_is_flagged_and_nothing_else_is_judged():
    unclassified = replace(item("Was there any argument about ports?", [Nugget("Sarah Hall made it.", V)]),
                           breadth="", min_points=None)
    assert codes(unclassified) == {"L0"}


@pytest.mark.parametrize("vital", [
    "Sarah Hall made that argument for public ownership of water.",
    "Sarah Bool, in the stamp duty debate.",
    "Yes: Rebecca Paul, on 15 September 2026.",
    "Only slightly: Lauren Edwards said it incorporates two amendments.",
    "The record shows Richard Tice arguing for fracking, not a Conservative reversal.",
    "Baroness Taylor of Stevenage said there was a great appetite for patient capital.",
])
def test_L1_a_member_named_in_a_vital_point_of_a_broad_item(vital):
    assert "L1" in codes(item("Was there any argument that households cannot change supplier?", [Nugget(vital, V)]))


@pytest.mark.parametrize("text,name", [("Gordon McKee said UEFA was proposing it.", "Gordon McKee"),
                                       ("Alison McGovern said social care was ignored.", "Alison McGovern"),
                                       ("Mairead O'Brien raised it, for the SDLP.", "Mairead O'Brien"),
                                       ("UEFA said it would complain.", None),
                                       ("The Minister said the system failed.", None)])
def test_member_names_with_inner_capitals_are_names_and_acronyms_and_roles_are_not(text, name):
    from goldset.lint import member_named

    assert member_named(text) == name


def test_L1_a_single_dated_instance_cannot_be_what_a_broad_undated_question_needs():
    i = item("my nan lost out on her pension, did anyone in parliament do anything",
             [Nugget("The Commons debated the ombudsman's report on women's state pension age on 16 May 2024.", V)])
    assert "L1" in codes(i)


@pytest.mark.parametrize("question,vital", [
    ("What did Helen Morgan object to in the medical training bill?", "Helen Morgan objected to clause 7(1)."),
    ("Who argued that the money for welfare comes out of somebody's pocket?", "Sir Ashley Fox argued it."),
    ("whos the mp thats always banging on about chalk streams", "Josh Dean raised the River Stort."),
    ("What did the policing minister say about the away fans ban?", "Sarah Jones, the Minister for Policing and Crime, "
     "acknowledged the concern."),
    ("Which benches did the MP defending stamp duty abolition sit on?", "The Conservative benches; Sir Mel Stride moved "
     "the motion."),
])
def test_L1_naming_is_allowed_when_the_question_names_the_member_asks_who_or_names_the_office(question, vital):
    assert "L1" not in codes(item(question, [Nugget(vital, V)]))


def test_L1_never_applies_to_a_narrow_item_or_an_optional_example():
    assert "L1" not in codes(item("Was there any argument about ports?", [Nugget("Sarah Hall made it.", V)],
                                  breadth="narrow"))
    assert "L1" not in codes(item("Was there any argument about ports?",
                                  [Nugget("An argument for public ports.", V, kind="point"),
                                   Nugget("Sarah Hall made it.", False, kind="example")]))


def test_L1_a_vital_example_on_a_broad_item_is_flagged_even_without_a_recognisable_name():
    assert "L1" in codes(item("What was said about ports?", [Nugget("the member for Dover made it", V,
                                                                     kind="example")]))


@pytest.mark.parametrize("vital", [
    "The passage does not record the Government accepting the figure.",
    "The quoted passage does not set out the Government's offer.",
    "Both speeches quoted are from the Westminster Hall debate on the water sector.",
    "Both quoted speeches are Conservative; the comparison cannot be made from them alone.",
    "No: the Conservative speech attacks it as a holiday tax.",
    "A count question, not a retrieval one; it needs the whole record, not a passage.",
    "They are different debates with different movers; any comparison is impressionistic.",
    "The Reform contribution quoted is Richard Tice's; Nigel Farage does not appear in it.",
    "None of the quoted contributions mentions it.",
])
def test_L2_a_vital_point_about_the_passage_or_the_method_not_the_record(vital):
    assert "L2" in codes(item("Who was more critical, the Conservatives or Labour?", [Nugget(vital, V)],
                              breadth="narrow"))


def test_L2_record_wording_is_fine():
    assert "L2" not in codes(item("Did the Government accept the figure?", [
        Nugget("Says plainly that the record found does not show the Government accepting the figure.", V,
               kind="scope")], breadth="narrow"))


@pytest.mark.parametrize("vital", [
    "1.8 billion pounds, as cited by Julia Lopez.",
    "Almost 89,000, as claimed by Rebecca Paul.",
    "£60.5 billion in gross value added in 2023, according to Sureena Brackenridge.",
    "A triathlon: Gregory Campbell (DUP) said Jewish athletes were forced to withdraw.",
    "No: Kate Dearden, Minister for the Future of Work, called the argument a false choice.",
    "Hammersmith and Fulham council, which recognises it as a protected characteristic (Ben Coleman, Chelsea and "
    "Fulham).",
])
def test_L3_a_fact_bundled_with_an_unasked_citer(vital):
    assert "L3" in codes(item("How much did the OBR say digital ID would cost?", [Nugget(vital, V)], breadth="narrow"))


def test_L3_the_citer_is_fine_when_the_question_asks_whose_figure_it_was():
    assert "L3" not in codes(item("Who cited the OBR's cost of digital ID?",
                                  [Nugget("1.8 billion pounds, as cited by Julia Lopez.", V)], breadth="narrow"))


def test_L3_a_fact_split_from_its_citer_is_fine():
    assert not codes(item("How much did the OBR say digital ID would cost?", [
        Nugget("1.8 billion pounds (the OBR's figure).", V, kind="fact"),
        Nugget("Julia Lopez cited it.", False, kind="example")], breadth="narrow"))


@pytest.mark.parametrize("question", ["What did both sides say about fracking?",
                                      "What did peers say for and against the two-child limit being lifted?",
                                      "What competing views did MPs give on Rosebank?"])
def test_L4_both_sides_needs_a_vital_point_on_each_side(question):
    one_sided = item(question, [Nugget("An argument for.", V, kind="point"),
                                Nugget("Another argument for.", V, kind="point")], min_points=2)
    assert "L4" in codes(one_sided)
    two_sided = item(question, [Nugget("An argument for.", V, kind="point", side="for"),
                                Nugget("An argument against.", V, kind="point", side="against")], min_points=2)
    assert "L4" not in codes(two_sided)


def test_L5_a_broad_item_needs_min_points():
    assert "L5" in codes(item("What was said about ports?", [Nugget("An argument for ports.", V, kind="point")],
                              min_points=None))


# --- recall on the measured class, precision on well-formed items --------------------------------------------------


def test_the_lint_catches_every_measured_offender_under_its_own_rule():
    missed = []
    for row in FROZEN["items"]:
        frozen = replace(from_dict(row["item"]), breadth=row["breadth"])
        if frozen.breadth == "broad" and frozen.min_points is None:
            frozen = replace(frozen, min_points=1)  # so L5 cannot be what catches it
        if row["audit_class"] not in codes(frozen):
            missed.append((frozen.id, row["audit_class"], sorted(codes(frozen))))
    assert not missed, missed
    assert len(FROZEN["items"]) == 32


WELL_FORMED = [
    item("What did both sides say about fracking and whether it is safe?", [
        Nugget("An argument that fracking can be done safely.", V, kind="point", side="for"),
        Nugget("An argument that it risks contaminating groundwater.", V, kind="point", side="against"),
        Nugget("Richard Tice pointed to millions living near US pads.", False, kind="example", side="for")],
        min_points=2),
    item("Did anyone actually use the words 'holiday tax' in the Commons?", [
        Nugget("Yes: members used the words 'holiday tax' in the Commons.", V, kind="fact"),
        Nugget("Rebecca Paul, on 15 September 2026.", False, kind="example")], breadth="narrow"),
    item("How much did the OBR say digital ID would cost?", [
        Nugget("1.8 billion pounds.", V, kind="fact"), Nugget("Julia Lopez cited it.", False, kind="example")],
        breadth="narrow"),
    item("What did Helen Morgan object to in the medical training bill?", [
        Nugget("That clause 7(1) would let Ministers change eligibility by the negative procedure.", V, kind="fact")],
        breadth="narrow"),
    item("Who raised the Devon reorganisation petition and which party are they?", [
        Nugget("Rebecca Smith raised it.", V, kind="fact"), Nugget("She is a Conservative.", V, kind="fact")],
        breadth="narrow"),
    item("Do new residential developments put extra strain on doctors' surgeries?", [
        Nugget("An argument that new housing makes it harder to see a GP.", V, kind="point"),
        Nugget("Lewis Cocking said so.", False, kind="example")], min_points=1),
    item("Tell me the cost of digital ID, who says so, and whether the Government accepts it.", [
        Nugget("1.8 billion pounds.", V, kind="fact"), Nugget("The OBR's figure, cited in the Commons.", V, kind="fact"),
        Nugget("Says plainly that the record found does not show the Government accepting the figure.", V,
               kind="scope")], breadth="narrow", expected_behaviour="partial", category="multi_part"),
    item("What did the Minister say about deporting foreign national offenders?", [
        Nugget("That the challenges in the Immigration Act 1971 are not merely technical.", V, kind="point")],
        breadth="narrow"),
    item("What concerns were raised about accountability for alleged war crimes?", [
        Nugget("A concern about how allegations could be addressed so those responsible face justice.", V, kind="point"),
        Nugget("Joe Morris asked it.", False, kind="example")], min_points=1),
    item("When was the away fans ban at Villa Park?", [Nugget("6 November.", V, kind="fact")], breadth="narrow"),
]


def test_the_lint_passes_well_formed_items():
    assert [(i.question[:40], [f.code for f in lint(i)]) for i in WELL_FORMED if lint(i)] == []


def test_lint_all_reports_every_finding_with_its_item_and_rule():
    findings = lint_all([WELL_FORMED[0], replace(WELL_FORMED[0], id="hum9002", min_points=None)])
    assert findings == [Finding("hum9002", "L5", findings[0].message)]
    assert "min_points" in findings[0].message


def test_dev_a_lints_clean():
    """S01b's exit bar: every dev-A item states its breadth and breaks none of L1 to L5. A new dev item, or an edit,
    that breaks a rule fails here before it can be judged.
    """
    from goldset.schema import load

    dev = [i for i in load() if i.split == "dev"]
    assert len(dev) == 223
    assert [(f.item, f.code, f.message[:90]) for f in lint_all(dev)] == []
