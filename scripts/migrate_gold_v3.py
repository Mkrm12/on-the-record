"""Migrate dev-A to gold schema v3."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goldset.lint import asks_who, member_named  # noqa: E402
from goldset.schema import GOLD_PATH, content_hash, from_dict, validate, vital_required  # noqa: E402
from goldset.split import split_hash  # noqa: E402

DATE = "2026-10-10"
TAG = f"[v3 {DATE}]"
# What the first application changed, as it stood before: the review page's "before", and the bank's check that a
# sentinel's verdict still holds (`gemini_judge.verdict_carries_over`).
BEFORE_PATH = ROOT / "tests" / "data" / f"gold_v3_dev_before_{DATE}.json"
REVIEW_PATH = ROOT / "reports" / f"gold_v3_dev_review_{DATE}.md"
REPAIRED = (f"{DATE}: schema v3 for dev-A (breadth on every item, a kind on every nugget; 32 rewrites for D-052; "
            "D-028 hum0083, D-062 db0237, hum0095 re-typed answer)")


class Refused(Exception):
    """The migration will not write: it was asked to edit outside dev-A, or the result does not validate."""


# --- the speeches the rewrites cite (all outside the sealed window) ------------------------------------------------

SARAH_HALL = "C0D2A8C8-52D0-4B57-A3DD-5AE554DE52F2"
SARAH_BOOL = "AFDCFA87-E050-486A-9B3C-5FD0B52E53AF"
JULIA_LOPEZ = "5003558C-5263-4847-BB68-7D86CF12814D"
REBECCA_PAUL = "5C8CFAA5-F430-49A7-989B-7934BD052D36"
RICHARD_TICE = "9BEA6E54-028C-44C0-B169-57B97A76BF1E"
CLAIRE_YOUNG_FRACKING = "4D4A4F2B-159C-4346-9F48-0265841985A6"
LIZZI_COLLINGE = "070AC1AD-758B-4DDA-9BE4-55530270027B"
ASHLEY_FOX = "174220A0-6F8D-441F-9DA2-5ABC77CC86B4"
SAQIB_BHATTI = "5C530750-9B00-4C1A-867D-1B7810720AC1"
LUKE_EVANS = "EED20239-B060-4931-9157-6134AACD4301"
KARL_TURNER = "F22B37BB-4057-4A2B-9702-C9976D737701"
ANDY_MCDONALD = "A3FE5A2F-4EE7-4C7D-B757-353757960AEB"
SARAH_RUSSELL = "0C7AEDCA-8B70-4E66-849C-494F3B0A0936"
PAM_COX = "5265C1E9-48ED-4F37-A450-B5BB3E79D11C"
STAMP_DUTY_MOTION = "5B961362-CA26-409D-8A75-AA1F4F797E67"
MARK_PRITCHARD = "E1E321B0-6A07-4264-963A-8846A6E6DABC"
GASCOIGNE = "AF41CB09-0D7A-4BA1-A75C-44685A6CBAD9"
PAUL_DAVIES = "7A777EBB-1E79-44E4-A291-617EFD0A22C1"
GREGORY_STAFFORD = "87955B03-4D42-421C-9E12-1E355F8769AE"
CLAIRE_YOUNG_WATER = "E1C98DAC-2554-4EB6-9407-3FA04891A9E1"
MANUELA_PERTEGHELLA = "75F32D9E-8691-44B1-B602-839B654CA784"
EMMA_HARDY = "C5510DD5-BD38-4404-BDCA-BC1D399E9242"
NIGEL_HUDDLESTON = "9E454478-93EC-4322-9DAE-342C6B521A7A"
JONATHAN_BRASH = "34DDFEBB-2854-43C0-BD7D-03868A74945C"
HARRIET_CROSS_MARKET = "A2B4F38B-C3C7-4D66-A2AD-352FF56DEAC1"
HARRIET_CROSS_NORTH_SEA = "114F1E4D-5F12-419B-AC93-F0D30B82C358"
LORD_COLLINS = "A4BC4CE6-2282-4ADA-8BCC-3D5FF326F6B1"
DIGITAL_ID_MINISTER = "4FDF4791-2461-4728-8195-2A3B25F960F9"
CHRIS_STEPHENS = "DDB1530A-91D0-4171-AE23-27A6B8C3DDBD"
JOHN_GLEN_STATEMENT = "70571D22-F0C3-4DCE-835D-805991EA0797"
SPEAKER_BEFORE_STATEMENT = "000599A3-8F23-44F7-8BCF-A5944ADD8C3A"
KEMI_BADENOCH = "FE53CA31-BF1E-4A5A-80B2-73DF66A650A5"
WENDY_MORTON = "6611FF53-A77E-4A0E-B67D-0C19E2A2AF3F"
ANDREW_ROSINDELL = "3F9B31C3-E2D2-48A3-9593-CEA61D02C5E7"
GREGORY_CAMPBELL = "3B3149C2-18BA-4158-802B-6EDC60DE0D7F"
LEWIS_COCKING = "B6C0A1AC-2A97-408F-A159-2825D1C583AB"
BARONESS_TAYLOR = "9400ABDD-23C6-4777-B3A2-0DDFF5A45E91"
KATE_DEARDEN = "771A43D3-903C-4946-A60D-834D2EBB5FD9"
GORDON_MCKEE = "C57FBA74-643E-418C-9077-586F50A3912D"
BEN_COLEMAN = "832ADCCC-8F5F-4962-86EA-1A3E448C65DE"
SUREENA_BRACKENRIDGE = "6DC7798A-474C-4116-B896-A99BBF725D3F"
AMANDA_MARTIN = "3BDE03B9-336E-4C5F-80BF-524C9E42553E"
ALAN_CAMPBELL = "39B6389A-FC1C-478E-9C8F-0BB09642EB53"
CATHERINE_FOOKES = "36C471F6-4590-4217-8828-FF6CDEC89C0E"
STEVE_YEMM = "331A28E8-4A53-416D-92CE-A0FE3A640019"
CAPITA_ANSWER = "2304E462-93A2-46CC-813E-E0F5C20298F4"
CAPITA_STATEMENT = "62B1F940-AC9E-4089-BB6E-C63A31F90164"
ALICIA_KEARNS = "09DE5DE0-3EC0-4E42-9194-31B9260F461F"
JOE_MORRIS = "8042F811-6881-4F89-9D6D-67BDEC3DADC2"
MEG_HILLIER = "C5B605D3-4DAD-443B-85E9-DB68B09F18D8"
LAUREN_EDWARDS = "32A6B50A-9FBD-4811-88F1-41C5E8113BDD"
YOUTH_HUBS = "0D69B72B-DE38-4818-9A24-CF3D36286E38"
LORD_BEST = "B69C8940-AB0A-4480-AD87-E8E9FCE9B9CE"
BARONESS_SCOTT = "0AE54D98-00A5-497B-BE2B-9DD37E9B74B6"
LORD_YOUNG = "73EDB11C-E91B-44E1-9B1A-EF1FD1E33327"
GREG_SMITH = "66D8DADC-061C-4602-AFD9-4F4AB0E0B6B3"


def N(text: str, vital: bool = False, kind: str = "fact", side: str = "", ids: tuple[str, ...] = ()) -> dict:
    """A nugget in gold_v2.json's own shape (schema._nugget_dict): kind and side only when set."""
    out: dict[str, Any] = {"text": text, "vital": vital, "speech_ids": list(ids)}
    if kind:
        out["kind"] = kind
    if side:
        out["side"] = side
    return out


# --- breadth and kinds for every dev item --------------------------------------------------------------------------

# The broad dev-A items: their answer space is the record, not one fact, member or event.
BROAD = frozenset({"hum0011", "hum0018", "hum0046", "hum0048", "hum0050", "hum0062", "hum0063", "hum0065", "hum0068",
                   "hum0072", "hum0076", "hum0079", "hum0081", "hum0087", "old0002", "old0008", "old0009", "old0019",
                   "old0021", "old0022"})
# Categories whose vital nuggets are what someone argued, not a figure, date or name.
POINT_CATEGORIES = frozenset({"named_speaker", "stance", "paraphrase", "multi_speech", "loaded"})
# (item, nugget index) -> kind, where the rule below is wrong for that nugget.
KIND_OVERRIDES = {("hum0017", 0): "fact",  # the answer is a peer's name: the question asks "did a peer..."
                  ("hum0033", 0): "fact",  # two figures (36 years, 33 years)
                  ("old0003", 0): "point",  # what the Minister said, not a figure
                  ("old0004", 1): "example",  # "Angela Rayner gave the figure": 'gave' is not one of lint's verbs
                  ("adv0059", 1): "scope",  # the framing is the questioner's
                  ("adv0061", 0): "fact",
                  ("adv0061", 1): "scope"}  # fmt: skip


def default_kind(item: dict, nugget: dict) -> str:
    """The kind a nugget of an item this migration does not rewrite gets."""
    behaviour = item["expected_behaviour"]
    if behaviour.startswith("abstain") or behaviour == "clarify_ambiguous":
        return "scope"
    if not nugget["vital"] and member_named(nugget["text"]):
        return "example"
    if (nugget["vital"] and item["provenance"] != "db_template" and item["category"] in POINT_CATEGORIES
            and not asks_who(item["question"])):
        return "point"
    return "fact"


# --- the rewrites --------------------------------------------------------------------------------------------------

# id -> what v3 sets.
REWRITES: dict[str, dict[str, Any]] = {
    # ---- L1: a member or one instance was vital on a broad question --------------------------------------------
    "hum0011": {
        "nuggets": [N("Yes: it was argued, for public ownership of water, that people cannot switch to another water "
                      "company when they are unhappy.", True, "point", ids=(SARAH_HALL,)),
                    N("Sarah Hall made the argument in Westminster Hall on 14 September 2026.", False, "example",
                      ids=(SARAH_HALL,))],
        "min_points": 1,
        "note": "L1: the argument is the vital point; Sarah Hall is an optional example (D-052).",
        "_why": "L1: the vital named Sarah Hall; the question asks whether the argument was made at all.",
        "_checked": "C0D2A8C8 (Sarah Hall, 14 Sep 2026): 'cannot suddenly decide ... switch to another water company'.",
    },
    "hum0018": {
        "nuggets": [N("Yes: the complaint that younger people are stuck renting, unable to buy, while older owners "
                      "cannot move.", True, "point", ids=(SARAH_BOOL,)),
                    N("Sarah Bool made it in the stamp duty debate on 28 October 2025.", False, "example",
                      ids=(SARAH_BOOL,))],
        "min_points": 1,
        "note": "L1: the complaint is the vital point; Sarah Bool is an optional example (D-052).",
        "_why": "L1: the vital was 'Sarah Bool, in the stamp duty debate'; the question asks whether the complaint was made.",
        "_checked": "AFDCFA87 (Sarah Bool, 28 Oct 2025): 'younger people stuck at the gatehouse, renting...'.",
    },
    "hum0046": {
        "nuggets": [N("For: the safety fears are overstated: millions of people in the United States live near "
                      "fracking pads, a shale well is barely noticeable once drilled, and the tremors involved are "
                      "tiny.", True, "point", "for", ids=(RICHARD_TICE,)),
                    N("Against: the risks are real: earthquakes, contamination of the aquifers families and farms "
                      "depend on, and more emissions.", True, "point", "against", ids=(CLAIRE_YOUNG_FRACKING,)),
                    N("Richard Tice (Reform UK) pointed to the 20 million Americans living within a mile of fracking "
                      "pads.", False, "example", "for", ids=(RICHARD_TICE,)),
                    N("Claire Young (Liberal Democrat) said the evidence of risk was in the Government's own records, "
                      "citing the Preston New Road earthquake.", False, "example", "against",
                      ids=(CLAIRE_YOUNG_FRACKING,))],
        "min_points": 2,
        "note": "L1/L4: a point per side, members as optional examples. Also corrects the old vital, which said Claire "
                "Young called the evidence base thin: she said the evidence of risk is there (D-052).",
        "_why": "L1: both vitals named members; the question asks what both sides said.",
        "_checked": "9BEA6E54 (Tice) and 4D4A4F2B (Claire Young), 15 Sep 2026, read in full.",
    },
    "hum0048": {
        "question": "What did MPs say for and against lifting the two-child limit?",
        "nuggets": [N("For lifting it: child poverty is not the children's fault, and it is unfair to credit wealth to "
                      "hard work while blaming poverty on the poor.", True, "point", "for", ids=(LIZZI_COLLINGE,)),
                    N("Against: benefits are paid by taxpayers, now or later if borrowed, and a fair system must "
                      "balance support with the needs of those who pay for it.", True, "point", "against",
                      ids=(ASHLEY_FOX,)),
                    N("Lizzi Collinge (Labour) argued for lifting it at the Bill's second reading on 3 February "
                      "2026.", False, "example", "for", ids=(LIZZI_COLLINGE,)),
                    N("Sir Ashley Fox (Conservative) argued for keeping it.", False, "example", "against",
                      ids=(ASHLEY_FOX,))],
        "forbidden_claims": [],
        "min_points": 2,
        "note": "The question said 'peers' as a scoping trap, but the record has peers on the two-child limit (e.g. 7 "
                "Lords speeches on 22 July 2024), so the trap's premise was false and the vitals (two MPs) answered a "
                "different question. The question now asks about MPs, with a point per side (D-052).",
        "_why": "L1, and a false trap: the record does have peers on the two-child limit.",
        "_checked": "Lords speeches mentioning the two-child limit outside the sealed window: 2020 and July 2024; "
                    "inside it, 32 speeches in 12 debates (counted, not read).",
    },
    "hum0072": {
        "nuggets": [N("Peers pressed for property agents to be regulated, with amendments that would create a regulator "
                      "for managing agents.", True, "point", ids=(LORD_BEST, LORD_YOUNG)),
                    N("The Government said the Bill's measures would make managing agents more accountable, and "
                      "declined a regulator for now, wanting standards for all agents dealt with in the round, not "
                      "piecemeal.", True, "point", ids=(BARONESS_SCOTT, GASCOIGNE)),
                    N("Lord Best (crossbench) said the case for a property agent regulator was now unavoidable (1 May "
                      "2024).", False, "example", ids=(LORD_BEST,)),
                    N("Lord Gascoigne replied for the Government on 24 May 2024, to amendments from Lord Best, "
                      "Baroness Hayter, Lord Bailey of Paddington and Lord Young.", False, "example",
                      ids=(GASCOIGNE,)),
                    N("Baroness Scott of Bybrook replied for the Government on 1 May 2024.", False, "example",
                      ids=(BARONESS_SCOTT,))],
        "min_points": 1,
        "note": "L1: what was said is the vital point, from either side of the exchange (the Lords debated it on 1 and "
                "24 May 2024); the members are optional examples (D-052).",
        "_why": "L1: the vital was 'Lord Gascoigne responded to amendments', which says who, not what.",
        "_checked": "AF41CB09 (Lord Gascoigne, 24 May 2024), read in full; B69C8940 (Lord Best) and 0AE54D98 "
                    "(Baroness Scott of Bybrook), 1 May 2024; 73EDB11C (Lord Young of Cookham), 24 May 2024.",
    },
    "hum0079": {
        "nuggets": [N("Yes: several members used the phrase 'holiday tax' in the Commons.", True, "fact"),
                    N("Rebecca Paul called the levy 'Labour's holiday tax' on 15 September 2026.", False, "example",
                      ids=(REBECCA_PAUL,)),
                    N("Nigel Huddleston used it on 3 September 2026, and Jonathan Brash on 25 March 2026.", False,
                      "example", ids=(NIGEL_HUDDLESTON, JONATHAN_BRASH))],
        "min_points": 1,
        "note": "L1: 'did anyone' is answered by any member who used the phrase; six did in the record (D-052).",
        "_why": "L1: the vital was 'Rebecca Paul, on 15 September 2026'; the question asks whether anyone did.",
        "_checked": "Commons speeches with 'holiday tax' outside the sealed window: Brash (25 Mar 2026), Huddleston and "
                    "Wild (3 Sep), Glen, Vickers and Paul (15 Sep).",
    },
    "hum0081": {
        "nuggets": [N("The record does not show a Conservative U-turn on fracking.", True, "scope"),
                    N("In the September 2026 debate on banning fracking, the Conservative spokesperson left onshore "
                      "fracking to the market and pressed for more North Sea gas instead.", False, "point",
                      ids=(HARRIET_CROSS_MARKET, HARRIET_CROSS_NORTH_SEA)),
                    N("Richard Tice (Reform UK) argued for fracking in that debate.", False, "example",
                      ids=(RICHARD_TICE,))],
        "min_points": 1,
        "note": "L1: the vital named Richard Tice (Reform), who cannot show a Conservative position either way. Verified "
                "outside the sealed window, where three Conservative speeches mention fracking and none reverses a "
                "position; two more inside the window were counted, not read (D-052).",
        "_why": "L1: the vital named a Reform MP for a question about the Conservatives.",
        "_checked": "Conservative fracking mentions outside the sealed window: Lord Hannan (18 Jul 2024), Lord "
                    "Blencathra (13 Jan 2026), Harriet Cross (15 Sep 2026, read in full). Inside: 2, counted.",
    },
    "hum0087": {
        "nuggets": [N("Yes: Parliament debated the Ombudsman's findings on how women born in the 1950s were told about "
                      "the rise in their state pension age, with calls for compensation.", True, "point"),
                    N("The Commons debated the Ombudsman's report on 16 May 2024.", False, "example")],
        "min_points": 1,
        "note": "L1: one dated debate was vital on an undated question; it is now an example (D-052).",
        "_why": "L1: a single dated instance was the only vital point on an undated question.",
        "_checked": "3F103EDB (16 May 2024): the debate on the Ombudsman's report and a compensation scheme.",
    },
    "old0002": {
        "nuggets": [N("That it means paying to give away sovereign British territory and then lease back a base the UK "
                      "already has.", True, "point", ids=(WENDY_MORTON, ANDREW_ROSINDELL)),
                    N("That the money is needed at home: it was set against cuts, such as the money to train British "
                      "troops.", True, "point", ids=(KEMI_BADENOCH,)),
                    N("Kemi Badenoch, Leader of the Opposition, asked at Prime Minister's Questions on 9 September "
                      "2026 why the Government was choosing to send £35 billion to Mauritius when it had cut the money "
                      "to train the troops.", False, "example", ids=(KEMI_BADENOCH,)),
                    N("Wendy Morton said the Government was paying Mauritius £35 billion to lease back a base the UK "
                      "already had (20 October 2025).", False, "example", ids=(WENDY_MORTON,)),
                    N("Andrew Rosindell called it a reckless decision to surrender £35 billion of taxpayers' money (28 "
                      "October 2025).", False, "example", ids=(ANDREW_ROSINDELL,)),
                    N("The £35 billion is the Opposition's figure, asserted in debate, not an established cost.", False,
                      "scope", ids=(KEMI_BADENOCH,))],
        "min_points": 1,
        "note": "L1: several opposition MPs objected, for two main reasons; either is a right answer, and the members "
                "are examples (D-052).",
        "_why": "L1: the only vital named Kemi Badenoch; the question asks why 'an opposition MP' objected.",
        "_checked": "Commons speeches with Mauritius and £35 billion outside the sealed window: Morton, Stuart, Patel "
                    "(20 Oct 2025), Rosindell (28 Oct, 2 Dec 2025), Badenoch (9 Sep 2026).",
    },
    "old0008": {
        "nuggets": [N("Yes, as a concern raised in debate: new housing makes it much harder for people to see their GP.",
                      True, "point", ids=(LEWIS_COCKING,)),
                    N("Lewis Cocking raised it on 7 September 2026.", False, "example", ids=(LEWIS_COCKING,)),
                    N("It is a member's concern, not an established finding.", False, "scope", ids=(LEWIS_COCKING,))],
        "min_points": 1,
        "note": "L1: the concern is the vital point; the member is an optional example (D-052).",
        "_why": "L1: the vital named Lewis Cocking; the question asks whether developments strain surgeries.",
        "_checked": "B6C0A1AC (Lewis Cocking, 7 Sep 2026).",
    },
    "old0009": {
        "nuggets": [N("A minister reported a great appetite among City investors for patient capital investment in "
                      "social housing.", True, "point", ids=(BARONESS_TAYLOR,)),
                    N("Baroness Taylor of Stevenage said so of a workshop she held in the City (15 September 2026).",
                      False, "example", ids=(BARONESS_TAYLOR,))],
        "min_points": 1,
        "note": "L1: the reported appetite is the vital point; the minister is an optional example (D-052).",
        "_why": "L1: the vital named Baroness Taylor; the question asks whether the enthusiasm exists.",
        "_checked": "9400ABDD (Baroness Taylor of Stevenage, 15 Sep 2026).",
    },
    "old0019": {
        "nuggets": [N("Against: it would not take a penny off bills and would blow the carbon emissions targets.", True,
                      "point", "against", ids=(CATHERINE_FOOKES,)),
                    N("For: such fields bring jobs, investment, skills and tax revenue, and support a just "
                      "transition.", True, "point", "for", ids=(STEVE_YEMM,)),
                    N("Catherine Fookes opposed drilling at Rosebank (3 September 2026).", False, "example", "against",
                      ids=(CATHERINE_FOOKES,)),
                    N("Steve Yemm argued for Jackdaw and Rosebank.", False, "example", "for", ids=(STEVE_YEMM,)),
                    N("Kerry McCarthy is named in Catherine Fookes's speech; she is not its speaker.", False, "fact",
                      ids=(CATHERINE_FOOKES,))],
        "min_points": 2,
        "note": "L1/L4: a point per side, members as optional examples (D-052).",
        "_why": "L1: both vitals named members; the question asks for the competing views.",
        "_checked": "36C471F6 (Catherine Fookes) and 331A28E8 (Steve Yemm), 3 Sep 2026.",
    },
    "old0021": {
        "nuggets": [N("How allegations of war crimes, in Gaza, the West Bank and southern Lebanon, can be addressed and "
                      "the evidence compiled so that those responsible are brought to justice.", True, "point",
                      ids=(JOE_MORRIS,)),
                    N("That the Government had closed the atrocity prevention centre set up in the Cabinet Office.",
                      True, "point", ids=(ALICIA_KEARNS,)),
                    N("Joe Morris asked how the allegations could be addressed (14 September 2026).", False, "example",
                      ids=(JOE_MORRIS,)),
                    N("Alicia Kearns raised the closure of the centre.", False, "example", ids=(ALICIA_KEARNS,))],
        "min_points": 1,
        "note": "L1: each concern is a vital point, either one answers; the members are examples (D-052).",
        "_why": "L1: the vital named Joe Morris; the question asks what concerns were raised.",
        "_checked": "8042F811 (Joe Morris) and 09DE5DE0 (Alicia Kearns), 14 Sep 2026.",
    },
    "old0022": {
        "nuggets": [N("Only slightly: it incorporates two amendments agreed without a vote in the House of Lords last "
                      "time.", True, "fact", ids=(LAUREN_EDWARDS,)),
                    N("Opponents said it was essentially the same Bill, though its Lords sponsor had said it needed 77 "
                      "amendments to be safer.", False, "point", ids=(MEG_HILLIER,)),
                    N("Lauren Edwards gave the first account and Meg Hillier the second (11 September 2026).", False,
                      "example", ids=(LAUREN_EDWARDS, MEG_HILLIER))],
        "min_points": 1,
        "note": "L1: the change itself is the vital fact; the opposing claim is context; members are examples (D-052).",
        "_why": "L1: both vitals named members; the question asks whether the Bill changed.",
        "_checked": "32A6B50A (Lauren Edwards) and C5B605D3 (Meg Hillier), 11 Sep 2026.",
    },
    # ---- L2: worded about the passage or the method -------------------------------------------------------------
    "hum0062": {
        "nuggets": [N("Both criticised the plans: Conservatives opposed restricting jury trials, and some Labour "
                      "backbenchers raised concerns too, while others defended the changes.", True, "point",
                      ids=(SAQIB_BHATTI, KARL_TURNER, ANDY_MCDONALD, SARAH_RUSSELL)),
                    N("Which group was more critical is a judgement the record does not settle.", True, "scope"),
                    N("For the Conservatives, Saqib Bhatti and Dr Luke Evans opposed the plans in the jury trials "
                      "debate of 7 January 2026.", False, "example", ids=(SAQIB_BHATTI, LUKE_EVANS)),
                    N("Among Labour backbenchers, Karl Turner, Andy McDonald and Sarah Russell raised concerns in "
                      "December 2025, while Pam Cox defended the changes.", False, "example",
                      ids=(KARL_TURNER, ANDY_MCDONALD, SARAH_RUSSELL, PAM_COX))],
        "forbidden_claims": ["A confident ranking of the two groups' criticism, as though the record measured it."],
        "min_points": 2,
        "note": "L2: the old vital said 'both quoted speeches are Conservative', which is true of the passage and false "
                "of the record: Labour backbench criticism is there (Oct-Dec 2025) (D-052).",
        "_why": "L2, and wrong at record level: the record does hold Labour backbench criticism.",
        "_checked": "Labour speeches on jury trials outside the sealed window, 14 Oct to 10 Dec 2025 (Vaughan, Johnson, "
                    "McDonald, Russell, Slaughter, Hamilton, Turner; Cox, Race and Foody defending).",
    },
    "hum0063": {
        "nuggets": [N("The record has no measure of harshness: the levy and the stamp duty proposals were debated "
                      "separately, so which reaction was harsher is a judgement.", True, "scope"),
                    N("Conservative members attacked the overnight visitor levy as a 'holiday tax'.", False, "point",
                      ids=(REBECCA_PAUL,)),
                    N("The Conservatives moved a motion to abolish stamp duty on primary residences on 28 October "
                      "2025.", False, "fact", ids=(STAMP_DUTY_MOTION,))],
        "min_points": 1,
        "note": "L2: the limit is stated about the record, not about the debates quoted (D-052).",
        "_why": "L2: 'any comparison is impressionistic' is method wording.",
        "_checked": "5C8CFAA5 (15 Sep 2026) and 5B961362 (28 Oct 2025).",
    },
    "hum0065": {
        "nuggets": [N("The record gives no single answer: it depends on what counts as talking about hospitality jobs. "
                      "Strictly (the phrase itself) 3 MPs did, against 6 who mentioned groundwater; loosely "
                      "(hospitality and jobs in one speech) 13 did.", True, "scope"),
                    N("The record runs to 17 September 2026, so September is counted to that date.", False, "scope")],
        "forbidden_claims": ["Stating counts as exact without saying how the members were counted."],
        "min_points": 1,
        "note": "L2: the old vital described the method ('a count question, not a retrieval one'). The counts are now "
                "computed: Commons, 1-17 September 2026, non-boilerplate speeches: 'hospitality jobs' 3 members, "
                "hospitality and job(s) in one sentence 6, anywhere in a speech 13; 'groundwater' 6 (D-052).",
        "_why": "L2: method wording; and the direction of the answer depends on the definition.",
        "_checked": "Counts by SQL, 2026-10-10 (see the note).",
    },
    "hum0068": {
        "nuggets": [N("No: Conservative members opposed it, attacking it as a 'holiday tax' and a tax on tourism.", True,
                      "point", ids=(REBECCA_PAUL, MARK_PRITCHARD)),
                    N("Rebecca Paul called it 'Labour's holiday tax' on 15 September 2026.", False, "example",
                      ids=(REBECCA_PAUL,)),
                    N("Mark Pritchard said it was not a visitor levy but a tax on tourism.", False, "example",
                      ids=(MARK_PRITCHARD,))],
        "min_points": 1,
        "note": "L2: 'the Conservative speech' becomes the Conservative position in the record (D-052).",
        "_why": "L2: the vital was about 'the Conservative speech', one passage.",
        "_checked": "Conservative speeches on the visitor levy outside the sealed window, Dec 2025 to Sep 2026: all "
                    "critical.",
    },
    "hum0076": {
        "nuggets": [N("Members argued for bringing water back into public ownership: privatisation had failed, and "
                      "customers cannot switch supplier.", True, "point", ids=(PAUL_DAVIES, SARAH_HALL)),
                    N("Some proposed other models, such as mutual ownership.", True, "point",
                      ids=(CLAIRE_YOUNG_WATER, MANUELA_PERTEGHELLA)),
                    N("Against nationalisation: it would not repair pipes and would move the financial risk, "
                      "potentially £100 billion, to taxpayers.", True, "point", ids=(GREGORY_STAFFORD,)),
                    N("The Minister said people were right to ask about ownership and promised reform and stronger "
                      "public control, without committing to public ownership.", True, "point", ids=(EMMA_HARDY,)),
                    N("Paul Davies opened the debate, on a petition for a referendum on public ownership (14 September "
                      "2026).", False, "example", ids=(PAUL_DAVIES,)),
                    N("Sarah Hall argued for public ownership.", False, "example", ids=(SARAH_HALL,)),
                    N("Claire Young and Manuela Perteghella proposed mutual ownership.", False, "example",
                      ids=(CLAIRE_YOUNG_WATER, MANUELA_PERTEGHELLA)),
                    N("Gregory Stafford argued against nationalisation.", False, "example", ids=(GREGORY_STAFFORD,)),
                    N("Emma Hardy replied for the Government.", False, "example", ids=(EMMA_HARDY,))],
        "min_points": 2,
        "note": "L2: the old vital ('both speeches quoted are from the debate') was not an answer. The points now come "
                "from the debate itself, Water Sector: Public Ownership, 14 September 2026; any two answer (D-052).",
        "_why": "L2: the vital described the passage and gave no answer.",
        "_checked": "Westminster Hall, 14 Sep 2026: speeches mentioning ownership (Davies, Gardiner, Gardner, Prinsley, "
                    "Perteghella, Young, Milne, Ribeiro-Addy, Burgon, Newbury, Dean, Stafford, Hardy).",
    },
    "hum0094": {
        "nuggets": [N("£1.8 billion.", True, "fact", ids=(JULIA_LOPEZ,)),
                    N("The OBR, as cited by Julia Lopez.", True, "fact", ids=(JULIA_LOPEZ,)),
                    N("The record does not show the Government accepting the figure: ministers said the cost could not "
                      "yet be given, or did not yet exist because the system had not been designed.", True, "scope",
                      ids=(DIGITAL_ID_MINISTER, LORD_COLLINS)),
                    N("Lord Collins of Highbury said on 18 March 2026 that there was no cost yet, because the system "
                      "had not been designed.", False, "example", ids=(LORD_COLLINS,))],
        "note": "L2: 'the passage does not record' becomes what the record shows, checked outside the sealed window. "
                "Three minister speeches inside the window mention the figure or the OBR and were counted, not read: "
                "re-check this item when W5 opens (D-052).",
        "_why": "L2: the scope was stated about the passage.",
        "_checked": "Minister speeches on digital ID and cost outside the sealed window: 14 Oct 2025 (met within the SR "
                    "budget), 18 Mar 2026 (no cost yet). None accepts £1.8bn.",
    },
    "hum0095": {
        "expected_behaviour": "answer",
        "nuggets": [N("Names a member who spoke about infected blood, with what that member said.", True, "point"),
                    N("Chris Stephens (SNP) spoke on 21 May 2024, with constituents in the gallery, and welcomed Sir "
                      "Robert Francis's appointment.", False, "example", ids=(CHRIS_STEPHENS,)),
                    N("The Government offered comprehensive compensation to infected and affected people, run by a new "
                      "Infected Blood Compensation Authority, with further interim payments of £210,000 to living "
                      "infected people.", True, "fact", ids=(JOHN_GLEN_STATEMENT,)),
                    N("John Glen, the Paymaster General, set out the scheme in a statement on 21 May 2024.", False,
                      "example", ids=(JOHN_GLEN_STATEMENT,))],
        "evidence_add": [{"speech_id": JOHN_GLEN_STATEMENT,
                          "quote": "Payments of £210,000 will be made to living infected beneficiaries"}],
        "note": "L2 and an objective correction: the old vital said the passage does not set out the Government's offer, "
                "but the record does (John Glen's statement, the same debate, 21 May 2024), so the item is answerable "
                "in full and is re-typed from partial to answer. 'Who spoke' has many true answers, so its vital point "
                "names a member and what they said; Chris Stephens is an example (A-03) (D-052).",
        "_why": "L2, and wrong at record level: the Government's offer is in the record.",
        "_checked": "70571D22 (John Glen's statement, 21 May 2024), read in full; DDB1530A (Chris Stephens).",
    },
    # ---- L3: a fact bundled with a citer the question did not ask about ----------------------------------------
    "hum0020": {
        "nuggets": [N("£1.8 billion, the OBR's estimate.", True, "fact", ids=(JULIA_LOPEZ,)),
                    N("Julia Lopez cited the OBR's figure on 8 December 2025.", False, "example", ids=(JULIA_LOPEZ,))],
        "note": "L3: the figure is split from the member who cited it (D-052).",
        "_why": "L3: '1.8 billion pounds, as cited by Julia Lopez'.",
        "_checked": "5003558C (Julia Lopez, 8 Dec 2025).",
    },
    "hum0023": {
        "nuggets": [N("Almost 89,000.", True, "fact", ids=(REBECCA_PAUL,)),
                    N("Rebecca Paul gave the figure for the Conservatives on 15 September 2026.", False, "example",
                      ids=(REBECCA_PAUL,))],
        "note": "L3: the figure is split from the member who claimed it (D-052).",
        "_why": "L3: 'Almost 89,000, as claimed by Rebecca Paul'.",
        "_checked": "5C8CFAA5 (Rebecca Paul, 15 Sep 2026).",
    },
    "old0006": {
        "nuggets": [N("A triathlon.", True, "fact", ids=(GREGORY_CAMPBELL,)),
                    N("Gregory Campbell (DUP) said Jewish athletes from Israel were forced to withdraw from it, blaming "
                      "'the hateful bile of these hard-left bigots' (8 September 2026).", False, "example",
                      ids=(GREGORY_CAMPBELL,)),
                    N("'Hostile protests' is the question's wording, not the record's: the record does not mention "
                      "protests here.", False, "scope", ids=(GREGORY_CAMPBELL,))],
        "note": "L3: the event is split from the member who described it (D-052).",
        "_why": "L3: 'A triathlon: Gregory Campbell (DUP) said...'.",
        "_checked": "3B3149C2 (Gregory Campbell, 8 Sep 2026).",
    },
    "old0007": {
        "nuggets": [N("A 'holiday tax': the overnight visitor levy.", True, "fact", ids=(NIGEL_HUDDLESTON,)),
                    N("Nigel Huddleston said the new Prime Minister had committed the Government to bringing in a new "
                      "holiday tax (3 September 2026).", False, "example", ids=(NIGEL_HUDDLESTON,)),
                    N("'Seasonal', 'tourists' and 'premier' are the question's words, not the record's.", False,
                      "scope", ids=(NIGEL_HUDDLESTON,))],
        "note": "L3: the charge is split from the member who reported it (D-052).",
        "_why": "L3: the vital bundled the levy with 'Nigel Huddleston said...'.",
        "_checked": "9E454478 (Nigel Huddleston, 3 Sep 2026, 'Overnight Visitor Levy: Rural and Coastal Communities').",
    },
    "old0010": {
        "nuggets": [N("No: the Government rejects the idea that workers' rights are bad for growth, calling it 'a false "
                      "choice'.", True, "point", ids=(KATE_DEARDEN,)),
                    N("Kate Dearden, Minister for the Future of Work, said so on 9 September 2026.", False, "example",
                      ids=(KATE_DEARDEN,))],
        "note": "L3: the Government's position is split from the minister who stated it (D-052).",
        "_why": "L3: 'No: Kate Dearden, Minister for the Future of Work, called...'.",
        "_checked": "771A43D3 (Kate Dearden, 9 Sep 2026).",
    },
    "old0011": {
        "nuggets": [N("UEFA: it was proposing a criminal complaint about the conduct of FIFA's leadership.", True, "fact",
                      ids=(GORDON_MCKEE,)),
                    N("Gordon McKee raised it on 3 September 2026.", False, "example", ids=(GORDON_MCKEE,))],
        "note": "L3: the body is split from the member who raised it (D-052).",
        "_why": "L3: 'UEFA: Gordon McKee said UEFA was proposing...'.",
        "_checked": "C57FBA74 (Gordon McKee, 3 Sep 2026).",
    },
    "old0012": {
        "nuggets": [N("Hammersmith and Fulham council, which recognises care experience as a protected characteristic.",
                      True, "fact", ids=(BEN_COLEMAN,)),
                    N("Ben Coleman (Chelsea and Fulham) said so on 3 September 2026.", False, "example",
                      ids=(BEN_COLEMAN,))],
        "note": "L3: the council is split from the member who named it (D-052).",
        "_why": "L3: the vital carried '(Ben Coleman, Chelsea and Fulham)'.",
        "_checked": "832ADCCC (Ben Coleman, 3 Sep 2026).",
    },
    "old0013": {
        "nuggets": [N("£60.5 billion in gross value added in 2023.", True, "fact", ids=(SUREENA_BRACKENRIDGE,)),
                    N("Including over £64 million in Wolverhampton.", False, "fact", ids=(SUREENA_BRACKENRIDGE,)),
                    N("Sureena Brackenridge gave the figure on 10 September 2026.", False, "example",
                      ids=(SUREENA_BRACKENRIDGE,))],
        "note": "L3: the figure is split from the member who gave it (D-052).",
        "_why": "L3: '... according to Sureena Brackenridge'.",
        "_checked": "6DC7798A (Sureena Brackenridge, 10 Sep 2026).",
    },
    "old0014": {
        "nuggets": [N("3.7 times the national average.", True, "fact", ids=(AMANDA_MARTIN,)),
                    N("Amanda Martin gave the figure on 8 September 2026.", False, "example", ids=(AMANDA_MARTIN,))],
        "note": "L3: the figure is split from the member who gave it (D-052).",
        "_why": "L3: '... according to Amanda Martin'.",
        "_checked": "3BDE03B9 (Amanda Martin, 8 Sep 2026).",
    },
    "old0015": {
        "nuggets": [N("11%.", True, "fact", ids=(ALAN_CAMPBELL,)),
                    N("With 63,000 knives taken off the streets.", False, "fact", ids=(ALAN_CAMPBELL,)),
                    N("Sir Alan Campbell, Leader of the House, gave the figures on 10 September 2026.", False, "example",
                      ids=(ALAN_CAMPBELL,)),
                    N("The record gives no baseline date for the 11%.", False, "scope", ids=(ALAN_CAMPBELL,))],
        "note": "L3: the figure is split from the minister who gave it (D-052).",
        "_why": "L3: '11%, according to Sir Alan Campbell, Leader of the House'.",
        "_checked": "39B6389A (Sir Alan Campbell, 10 Sep 2026).",
    },
    "old0020": {
        "nuggets": [N("It is withholding contractual payments from Capita.", True, "fact",
                      ids=(CAPITA_ANSWER, CAPITA_STATEMENT)),
                    N("It also deployed 147 surge staff, appointed an independent, Capita-funded remedial adviser, and "
                      "is assessing bringing the scheme in-house.", False, "fact", ids=(CAPITA_ANSWER, CAPITA_STATEMENT)),
                    N("Sally Jameson, Parliamentary Secretary at the Cabinet Office, set this out on 10 September 2026.",
                      False, "example", ids=(CAPITA_ANSWER, CAPITA_STATEMENT))],
        "note": "L3: the action is split from the minister who described it (D-052).",
        "_why": "L3: the vital carried '(Sally Jameson, Parliamentary Secretary, Cabinet Office)'.",
        "_checked": "2304E462 and 62B1F940 (10 Sep 2026).",
    },
    # ---- L3, found by the mechanical lint --
    "hum0077": {
        "nuggets": [N("No: it was given in the Lords.", True, "fact", ids=(YOUTH_HUBS,)),
                    N("Baroness Sherlock gave it on 24 March 2026.", False, "example", ids=(YOUTH_HUBS,))],
        "note": "L3, found by the lint on 2026-10-10: the House is split from the minister who spoke (D-052).",
        "_why": "L3: 'No: Baroness Sherlock said it in the Lords' bundles the fact with an unasked citer.",
        "_checked": "0D69B72B (Baroness Sherlock, Lords, 24 Mar 2026).",
    },
    "adv0015": {
        "nuggets": [N("Nothing: he is not a member and did not address the House; a member quoted his post.", True,
                      "fact", ids=(GREG_SMITH,)),
                    N("Greg Smith quoted Martin Brundle's post in the air traffic control debate on 10 September 2026.",
                      False, "example", ids=(GREG_SMITH,))],
        "note": "L3, found by the lint on 2026-10-10: the correction is split from the member who quoted him (D-052).",
        "_why": "L3: 'his post was quoted by Greg Smith MP' bundles the correction with an unasked citer.",
        "_checked": "66D8DADC (Greg Smith, 10 Sep 2026, 'Air Traffic Control Disruption'): the Brundle quotation.",
    },
    "adv0049": {
        "nuggets": [N("Nigel Farage did not speak in that debate; the Reform contribution was Richard Tice's.", True,
                      "scope", ids=(RICHARD_TICE,))],
        "note": "L2, found by the lint on 2026-10-10 ('the Reform contribution quoted'): now stated about the record, "
                "where Nigel Farage has no speech in the Fracking Ban debate of 15 September 2026 (D-052).",
        "_why": "L2: 'The Reform contribution quoted is Richard Tice's' is about the passage.",
        "_checked": "debate DE5D6B0F (Fracking Ban, 15 Sep 2026): no speech by Nigel Farage.",
    },
    # Over-specific narrow entries (named in the ledger; the lint has no rule for them) ----------------
    "hum0097": {
        "nuggets": [N("Yes: it comes up in more than one part of the record.", True, "fact"),
                    N("For example, it is debated in September 2026 and again in early 2026.", False, "fact")],
        "note": "D-029: 'more than once' was answerable only by naming two particular periods; any two parts of the "
                "record now answer it, and the periods are optional detail. The owner's triage verdict (J030) was given "
                "against the old wording, so the instrument bank holds this sentinel back until it is re-adjudicated.",
        "_why": "D-029: the vital demanded two particular periods; the question asks whether it recurs at all.",
        "_checked": "Local government reorganisation in the record: Dec 2025, Mar 2026, Sep 2026 (the dev s18 answer "
                    "and 0DCF0CDE).",
    },
    "hum0099": {
        "nuggets": [N("Yes: it appears in more than one part of the record.", True, "fact"),
                    N("For example, in late 2025, early 2026 and September 2026.", False, "fact")],
        "note": "D-029: as hum0097; the periods are optional detail.",
        "_why": "D-029: the vital demanded three particular periods; the question asks whether it recurs at all.",
        "_checked": "EDF06F9D (11 Sep 2026); the Bill's Lords stages in late 2025 and early 2026 (old0022's evidence).",
    },
    # ---- objective corrections ---------------------------------------------------------------------------
    "hum0083": {
        "nuggets": [N("Yes: the Government brought its own Bill to remove the limit, and the Commons gave it a second "
                      "reading on 3 February 2026, by 458 votes to 104.", True, "fact", ids=(LIZZI_COLLINGE,)),
                    N("It was a Government Bill, not a private Member's Bill.", False, "fact")],
        "forbidden_claims": ["Calling it a private Member's Bill: it was the Government's own.",
                             "Stating that the limit had already gone on 3 February 2026: the Bill was then at second "
                             "reading."],
        "note": "D-028, an objective correction (A-04): the Bills API types bill 4059, the Universal Credit (Removal of "
                "Two Child Limit) Bill, as a Government Bill (type 1), and it became an Act in 2026; the old vital "
                "called it a private Member's Bill and the old must-not-claim forbade saying the Government had "
                "abolished the limit. Division 2253 (second reading, 3 February 2026): agreed, 458 to 104.",
        "_why": "D-028: the Bills API says Government Bill; the old forbidden claim is contradicted by the Act.",
        "_checked": "bill.id=4059 (bill_type_id 1, is_act true); division.id=2253 (agreed 458-104); the audit's bill.type "
                    "check.",
    },
    "db0237": {
        "nuggets": [N("John Glen, the Paymaster General, opened it with the Government's statement.", True, "fact",
                      ids=(JOHN_GLEN_STATEMENT,)),
                    N("The Speaker spoke before him, but to preside, not to open the debate.", False, "scope",
                      ids=(SPEAKER_BEFORE_STATEMENT,))],
        "evidence": [{"speech_id": JOHN_GLEN_STATEMENT,
                      "quote": "With permission, Mr Speaker, I would like to make a statement following the final "
                               "report of the infected blood inquiry."}],
        "expected_structured_facts": {"debate_id": "8A74600F-6D18-4E73-A4C3-DFB1E59ED659",
                                      "heading": "Infected Blood Compensation Scheme", "opened_by": "John Glen",
                                      "mnis_id": 4051, "party": "Con"},
        "graded_relevance_set": {JOHN_GLEN_STATEMENT: 2, SPEAKER_BEFORE_STATEMENT: 0},
        "forbidden_claims": ["Any member who did not in fact open that debate.",
                             "Naming the Speaker as its opener: he presided."],
        "note": "D-062, an objective correction (A-04): the generator took the first non-boilerplate speech, which here "
                "was the Speaker's 'Before I call the Minister...'. D-001's definition of the opener excludes the "
                "Chair, so the opener is John Glen (mnis 4051), who made the statement.",
        "_why": "D-062: the gold credited the Speaker, who was presiding, with opening a ministerial statement.",
        "_checked": "debate 8A74600F: seq 2 the Speaker (presiding), seq 4 John Glen's statement; the audit's "
                    "who_spoke.opener check.",
    },
    # ---- sides only: a broad item that already met the rules on wording ----------------------------------------
    "hum0050": {
        "nuggets": [N("The Minister identified the Swanwick flight processing system and ruled out a cyber-attack.",
                      True, "point", "government"),
                    N("The shadow Minister put the disruption at over 2,000 flights and criticised the lack of a "
                      "statement.", True, "point", "opposition")],
        "min_points": 2,
        "note": "Breadth broad, a point per side (A-03); the wording is unchanged.",
        "_why": "Not an offender: it asks for both sides, so each side is marked.",
        "_checked": "Unchanged wording.",
    },
}


# --- the migration -------------------------------------------------------------------------------------------------


def migrated(item: dict) -> dict:
    """`item` as v3 wants it. A new dict; the same answer however many times it is applied."""
    out = copy.deepcopy(item)
    rewrite = REWRITES.get(item["id"], {})
    for field in ("question", "expected_behaviour", "nuggets", "forbidden_claims", "evidence",
                  "expected_structured_facts"):
        if field in rewrite:
            out[field] = copy.deepcopy(rewrite[field])
    for evidence in rewrite.get("evidence_add", ()):
        if evidence not in out["evidence"]:
            out["evidence"].append(copy.deepcopy(evidence))
    if "graded_relevance_set" in rewrite:
        out["graded_relevance"] = {**out.get("graded_relevance", {}), **rewrite["graded_relevance_set"]}
    out["breadth"] = "broad" if item["id"] in BROAD else "narrow"
    out.pop("min_points", None)
    if "min_points" in rewrite:
        out["min_points"] = rewrite["min_points"]
    if "nuggets" not in rewrite:
        for k, nugget in enumerate(out["nuggets"]):
            nugget["kind"] = KIND_OVERRIDES.get((item["id"], k)) or default_kind(out, nugget)
    note = rewrite.get("note")
    if note and f"{TAG} {note}" not in (out.get("notes") or ""):
        out["notes"] = " | ".join(x for x in (out.get("notes") or "", f"{TAG} {note}") if x)
    return out


def _refuse_outside_dev(by_id: dict[str, dict]) -> None:
    for item_id in REWRITES:
        item = by_id.get(item_id)
        if item is None:
            raise Refused(f"{item_id} is not in the gold set")
        if item["split"] != "dev":
            raise Refused(f"{item_id} is in the {item['split']} split: only dev-A is edited for this (C1)")


def check(path: Path = GOLD_PATH) -> list[str]:
    """What is not yet migrated. Empty means the migration is in place."""
    data = json.loads(path.read_text(encoding="utf-8"))
    problems = [] if data.get("schema_version") == 3 else [f"schema_version is {data.get('schema_version')}, not 3"]
    by_id = {i["id"]: i for i in data["items"]}
    try:
        _refuse_outside_dev(by_id)
    except Refused as exc:
        problems.append(str(exc))
    problems += [f"{i['id']} is not migrated" for i in data["items"] if i["split"] == "dev" and migrated(i) != i]
    items = [from_dict(i) for i in data["items"] if i.get("status") != "dropped"]
    if data.get("gold_hash") != content_hash(items) or data.get("split_hash") != split_hash(items):
        problems.append("the hashes do not name this instrument")
    return problems


def apply(path: Path = GOLD_PATH) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {i["id"]: i for i in data["items"]}
    _refuse_outside_dev(by_id)  # before anything is touched
    before, changed = {}, 0
    for item in data["items"]:
        if item["split"] != "dev":
            continue
        new = migrated(item)
        if new != item:
            before[item["id"]] = copy.deepcopy(item)
            item.clear()
            item.update(new)
            changed += 1
    data["schema_version"] = 3
    if errors := validate(data):
        raise Refused(f"the migrated set does not validate ({len(errors)}): {errors[:3]}")
    items = [from_dict(i) for i in data["items"] if i.get("status") != "dropped"]
    old_hash = data.get("gold_hash", "")
    data["gold_hash"], data["split_hash"] = content_hash(items), split_hash(items)
    if REPAIRED not in data.get("repaired", ""):
        data["repaired"] = "; ".join(x for x in (data.get("repaired", ""), REPAIRED) if x)
    if before and not BEFORE_PATH.exists():  # the first application keeps what it changed, for the review page
        BEFORE_PATH.write_text(json.dumps({"date": DATE, "items": before}, indent=1, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{changed} dev item(s) migrated; schema_version 3; gold_hash {old_hash} -> {data['gold_hash']}")
    return 0


# --- the maintainer's review page
# ---------------------------------------------------------------------------------------


def _nugget_lines(nuggets: list[dict]) -> list[str]:
    lines = []
    for k, n in enumerate(nuggets, start=1):
        tags = ", ".join(x for x in (n.get("kind", ""), f"side: {n['side']}" if n.get("side") else "") if x)
        lines.append(f"{k}. {'**[VITAL]** ' if n['vital'] else ''}{f'({tags}) ' if tags else ''}{n['text']}")
    return lines


def review_page(path: Path = GOLD_PATH) -> str:
    """Every rewrite, before and after, for the maintainer. Statuses stay provisional until this is reviewed."""
    now = {i["id"]: i for i in json.loads(path.read_text(encoding="utf-8"))["items"]}
    before = json.loads(BEFORE_PATH.read_text(encoding="utf-8"))["items"] if BEFORE_PATH.exists() else {}
    dev = [i for i in now.values() if i["split"] == "dev"]
    kinds = Counter(n.get("kind", "") for i in dev for n in i["nuggets"])
    out = [f"# Gold v3: the dev-A rewrites for review ({DATE})", "",
           f"Every item below is still `provisional`. {len(REWRITES)} items were rewritten; the other "
           f"{len(dev) - len(REWRITES)} dev items only gained a breadth and nugget kinds by rule "
           f"({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))}). Breadth: "
           f"{sum(i.get('breadth') == 'broad' for i in dev)} broad, {sum(i.get('breadth') == 'narrow' for i in dev)} "
           "narrow. Record claims were checked outside the sealed window.", "",
           "For each item: **approve**, **edit** (say what), or **reject** (it reverts).", ""]
    for item_id, rewrite in REWRITES.items():
        item, old = now[item_id], before.get(item_id, {})
        out += [f"## {item_id}", "",
                f"*{item['category']} · {item.get('breadth', '')}"
                + (f" · needs {vital_required(from_dict(item))} of {sum(n['vital'] for n in item['nuggets'])} vital"
                   if item.get("min_points") else "") + f" · {item['expected_behaviour']} · {item['status']}*", "",
                f"**Why:** {rewrite['_why']}", "", f"**Checked against the record:** {rewrite['_checked']}", ""]
        if old and old.get("question") != item["question"]:
            out += [f"**Question (before):** {old['question']}", "", f"**Question (now):** {item['question']}", ""]
        else:
            out += [f"**Question:** {item['question']}", ""]
        if old and old.get("expected_behaviour") != item["expected_behaviour"]:
            out += [f"**Behaviour:** {old['expected_behaviour']} → {item['expected_behaviour']}", ""]
        if old:
            out += ["**Before:**", "", *_nugget_lines(old["nuggets"]), ""]
        out += ["**Now:**", "", *_nugget_lines(item["nuggets"]), ""]
        if item.get("forbidden_claims") or old.get("forbidden_claims"):
            if old and old.get("forbidden_claims") != item.get("forbidden_claims"):
                out += ["**Must not claim (before):** " + ("; ".join(old.get("forbidden_claims") or []) or "none"), ""]
            out += ["**Must not claim:** " + ("; ".join(item.get("forbidden_claims") or []) or "none"), ""]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--review", action="store_true", help=f"write {REVIEW_PATH.relative_to(ROOT)}")
    p.add_argument("--path", type=Path, default=GOLD_PATH)
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if args.apply:
        try:
            return apply(args.path)
        except Refused as exc:
            print(f"REFUSED: {exc}")
            return 1
    if args.review:
        REVIEW_PATH.write_text(review_page(args.path), encoding="utf-8")
        print(f"wrote {REVIEW_PATH.relative_to(ROOT)}")
        return 0
    problems = check(args.path)
    for problem in problems:
        print(f"  {problem}")
    print(f"{len(problems)} problem(s)" if problems else "the v3 migration is in place")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
