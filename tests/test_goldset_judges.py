"""The judge panel and its rate limiting, and the blind W5 comparison."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from goldset import judges

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import w5_compare_graders as compare_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Every test in this file is hermetic: the developer's real `.env` must not decide whether a provider is
    available.
    """
    monkeypatch.setattr(judges, "_DOTENV_CACHE", {})


# --- the cross-family rule ------------------------------------------------------------------------------


def test_the_generators_own_family_can_never_be_the_primary_judge(monkeypatch):
    """: the answer-level judge comes from a different family from the generator. Enforced in code rather than
    remembered, because remembering it is how it gets forgotten.
    """
    monkeypatch.setenv("DEEPSEEK_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    available = judges.panel(exclude_family=judges.GENERATOR_FAMILY)
    assert "deepseek" not in [p.name for p in available]
    chosen = judges.primary(available, exclude_family=judges.GENERATOR_FAMILY)
    assert chosen is not None and chosen.family != "deepseek"


def test_a_fragile_provider_is_never_the_primary_judge(monkeypatch):
    """Gemini's free tier is tiny and its availability varies by country. It may take part; it may not be the
    authority, or one flaky member decides the project's numbers.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    for name in ("DEEPSEEK_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "CLAUDE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    available = judges.panel()
    assert [p.name for p in available] == ["google"]
    assert judges.primary(available) is None, "a fragile provider must not be promoted to authority"


def test_no_cross_family_judge_returns_none_rather_than_falling_back(monkeypatch):
    """The refusal that matters. A panel that quietly loses its cross-family member has stopped doing its job, so this
    returns None and makes the caller surface it.
    """
    monkeypatch.setenv("DEEPSEEK_KEY", "x")
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "CLAUDE_API_KEY", "GOOGLE_API_KEY", "GOOGLEAPIKEY",
                 "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    available = judges.panel(exclude_family="deepseek")
    assert available == []
    assert judges.primary(available, exclude_family="deepseek") is None


def test_the_anthropic_slot_exists_and_says_what_to_set(monkeypatch):
    """The maintainer's intended primary judge. No key is present, and that must be reported rather than papered over
    with a same-family substitute.
    """
    for name in judges.KEY_NAMES["anthropic"]:
        monkeypatch.delenv(name, raising=False)
    provider = judges.BY_NAME["anthropic"]
    assert provider.available is False
    assert "ANTHROPIC_API_KEY" in provider.unavailable_reason()
    assert "claude-3-5" not in provider.model, "use a current Claude 5 model, not one two generations old"


def test_a_key_is_found_under_the_spelling_people_actually_use(monkeypatch):
    """The maintainer's Google key is spelled `googleapikey`. A key that is present but ignored because of its variable
    name is the most annoying possible failure.
    """
    for name in ("GOOGLE_API_KEY", "GOOGLEAPIKEY", "GEMINI_API_KEY", "googleapikey"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("googleapikey", "abc")
    assert judges.BY_NAME["google"].key() == "abc"


def test_unavailable_reports_every_provider_without_a_key(monkeypatch):
    for names in judges.KEY_NAMES.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
            monkeypatch.delenv(name.lower(), raising=False)
    missing = judges.unavailable()
    assert set(missing) == set(judges.BY_NAME)
    assert all("set " in why for why in missing.values())


# --- rate limiting --------------------------------------------------------------------------------------


def test_the_limiter_sleeps_to_hold_the_minimum_interval():
    """A semaphore would allow a burst of four inside a second, which is what gets a free tier banned. The limit is
    requests per MINUTE, so the limiter serialises and sleeps.
    """
    slept: list[float] = []
    now = [0.0]

    async def fake_sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = judges.RateLimiter(min_interval_s=2.0, sleep=fake_sleep, clock=lambda: now[0])

    async def three():
        for _ in range(3):
            await limiter.acquire()

    asyncio.run(three())
    assert slept == [2.0, 2.0], "the first request goes straight through; the next two wait"
    assert limiter.used == 3


def test_the_limiter_does_not_sleep_when_enough_time_has_passed():
    slept = []
    now = [0.0]

    async def fake_sleep(seconds):
        slept.append(seconds)

    limiter = judges.RateLimiter(min_interval_s=1.0, sleep=fake_sleep, clock=lambda: now[0])
    asyncio.run(limiter.acquire())
    now[0] = 99.0
    asyncio.run(limiter.acquire())
    assert slept == []


def test_a_daily_cap_is_refused_before_the_request_not_after_a_ban():
    limiter = judges.RateLimiter(min_interval_s=0.0, per_day=2)

    async def three():
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()

    with pytest.raises(judges.QuotaExhausted, match="cap of 2"):
        asyncio.run(three())
    assert limiter.used == 2, "the refused request is not counted against the cap"


def test_no_daily_cap_means_no_exhaustion():
    limiter = judges.RateLimiter(min_interval_s=0.0, per_day=0)
    asyncio.run(limiter.acquire())
    assert limiter.exhausted is False


@pytest.mark.parametrize(("per_minute", "expected"), [(60, 1.0), (25, 2.4), (4, 15.0), (0, 0.0)])
def test_the_interval_comes_from_the_stated_limit(per_minute, expected):
    provider = judges.Provider("t", "t", "https://x", "m", per_minute=per_minute)
    assert provider.min_interval_s == pytest.approx(expected)


def test_groq_and_a_same_family_answer_judge_are_gone():
    """The maintainer's judge is Gemini, run offline (scripts/gemini_judge.py)."""
    assert "groq" not in judges.BY_NAME and "deepseek-judge" not in judges.BY_NAME
    assert all(p.family != judges.GENERATOR_FAMILY for p in judges.PROVIDERS if p.name != "deepseek"),         "DeepSeek is listed once, as E1's retrieval-relevance judge, and never as an answer judge"
    assert judges.BY_NAME["google"].fragile is True


# --- the panel ------------------------------------------------------------------------------------------


def test_a_split_panel_is_reported_rather_than_resolved():
    """Disagreement is the output. An item the panel splits on is one for the maintainer, and saying so is the point of
    running three families.
    """
    result = judges.PanelResult("q1", [
        judges.Verdict("a", "fa", "GRADE: 2"), judges.Verdict("b", "fb", "GRADE: 0"),
    ])  # fmt: skip
    got = result.agreement(lambda reply: reply.split(":")[-1].strip())
    assert got["split"] is True and got["unanimous"] is False
    assert sorted(got["distinct"]) == ["0", "2"]


def test_a_unanimous_panel_says_so():
    result = judges.PanelResult("q1", [judges.Verdict("a", "fa", "2"), judges.Verdict("b", "fb", "2")])
    got = result.agreement(lambda r: r)
    assert got["unanimous"] is True and got["split"] is False and got["answered"] == 2


def test_a_provider_that_failed_is_left_out_of_the_agreement():
    """A failure is not a vote. Counting it would let an outage look like a disagreement."""
    result = judges.PanelResult("q1", [
        judges.Verdict("a", "fa", "2"), judges.Verdict("b", "fb", None, error="timeout"),
    ])  # fmt: skip
    assert len(result.answered) == 1
    assert result.agreement(lambda r: r)["unanimous"] is True


def test_an_unparsable_reply_is_not_counted_as_a_label():
    result = judges.PanelResult("q1", [judges.Verdict("a", "fa", "nonsense"), judges.Verdict("b", "fb", "2")])
    got = result.agreement(lambda r: None if r == "nonsense" else r)
    assert got["distinct"] == ["2"] and got["unanimous"] is True


# --- the blind W5 comparison ----------------------------------------------------------------------------


def key_row(ref="w5-001", category="numeric_perturbation"):
    return {"ref": ref, "question": "Q?", "category": category, "criticality": "major",
            "my_reading": ["the figure is wrong"], "must_not_assert": ["the wrong figure"]}  # fmt: skip


def test_a_grader_who_says_the_passage_answers_an_adversarial_question_disagrees():
    """The disagreement worth the maintainer's time: if the passage really does answer it as asked, my item is wrong
    and a correct system would be scored as making a correction.
    """
    rows = compare_mod.compare([key_row()], {"w5-001": {"supports_answer": "yes", "contradiction": ""}})
    assert rows[0]["agrees"] is False and "DISAGREES" in rows[0]["verdict"]


@pytest.mark.parametrize("supports", ["no", "partly"])
def test_a_grader_who_finds_it_unsupported_agrees(supports):
    rows = compare_mod.compare([key_row()], {"w5-001": {"supports_answer": supports,
                                                        "contradiction": "the passage says 160,000"}})  # fmt: skip
    assert rows[0]["agrees"] is True and rows[0]["described_the_problem"] is True


def test_noticing_without_diagnosing_is_weaker_agreement_and_is_marked_as_such():
    rows = compare_mod.compare([key_row()], {"w5-001": {"supports_answer": "no", "contradiction": ""}})
    assert rows[0]["agrees"] is True
    assert rows[0]["described_the_problem"] is False


def test_a_loaded_item_is_judged_on_whether_the_framing_was_noticed():
    """A loaded question may be perfectly answerable; the test is whether the grader saw the framing."""
    spotted = compare_mod.compare([key_row(category="loaded")],
                                  {"w5-001": {"supports_answer": "yes", "loaded": "'disastrous' is the "
                                              "questioner's word"}})  # fmt: skip
    assert spotted[0]["agrees"] is True
    missed = compare_mod.compare([key_row(category="loaded")],
                                 {"w5-001": {"supports_answer": "yes", "loaded": ""}})  # fmt: skip
    assert missed[0]["agrees"] is False


def test_an_ungraded_item_is_reported_rather_than_counted_as_agreement():
    rows = compare_mod.compare([key_row()], {})
    assert rows[0]["graded"] is False and rows[0]["agrees"] is None


def test_the_worksheet_withholds_everything_except_the_allow_listed_fields():
    """Blind by allow-list, not by deletion: a field added to the seeds later is withheld by default rather than
    leaking because nobody remembered to remove it.
    """
    # The W5 tooling holds the sealed exam's items, so it exists only in the private checkout.
    worksheet = pytest.importorskip("w5_blind_worksheet")

    assert worksheet.WORKSHEET_FIELDS == ("ref", "question", "passages")
    for forbidden in ("category", "my_reading", "must_not_assert", "nug", "forbid", "crit"):
        assert forbidden not in worksheet.WORKSHEET_FIELDS


# --- keys must be findable where the maintainer actually keeps them --------------------------------------------


def test_a_key_in_dotenv_is_found_when_the_environment_has_none(monkeypatch, tmp_path):
    """The bug this pins. Nothing puts `.env` into the environment — `app.config` loads it through pydantic-settings
    into a `settings` object — so a provider reading only `os.environ` found no keys at all: measured, with
    DEEPSEEK_KEY and GROQ_KEY both present in the file and `panel` returning an empty list.
    """
    for name in judges.KEY_NAMES["google"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(judges, "_DOTENV_CACHE", {"google_api_key": "sk-from-the-file"})
    assert judges.Provider.__dict__  # the module imported
    google = next(p for p in judges.PROVIDERS if p.name == "google")
    assert google.key() == "sk-from-the-file"
    assert google.available is True


def test_the_environment_still_wins_over_the_file(monkeypatch):
    """An exported variable overrides the file, which is the direction every other tool assumes."""
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-exported")
    monkeypatch.setattr(judges, "_DOTENV_CACHE", {"google_api_key": "sk-from-the-file"})
    google = next(p for p in judges.PROVIDERS if p.name == "google")
    assert google.key() == "sk-exported"


def test_a_missing_dotenv_is_not_an_error(monkeypatch):
    monkeypatch.setattr(judges, "_DOTENV_CACHE", {})
    for name in judges.KEY_NAMES["google"]:
        monkeypatch.delenv(name, raising=False)
    google = next(p for p in judges.PROVIDERS if p.name == "google")
    assert google.key() == "" and google.available is False
