"""Card headers stated once per debate, every field kept — the maintainer's "headers are relevant, we need them" — and
the URL out of the model's view.
"""

from __future__ import annotations

import json
from datetime import date

from answer import cards as C
from answer import generate as G
from answer import prompts as PR
from answer.contract import Provenance
from ingest.runtime import run

PROV = Provenance(route="retrieval", resolved_filters={}, corpus_version="v2", retrieval_config_hash="c", recipe="r",
                  prompt_version="p", model_id="m")  # fmt: skip
URL = "https://hansard.parliament.uk/Commons/2026-09-15/debates/abc/Fracking#contribution-1"


def card(speech, debate="D1", speaker="Alison Hume", party="Lab", office=None, day=date(2026, 9, 15), house="Commons",
         section="Westminster Hall", heading="Fracking", chunk=1):
    return C.Card(speech_id=speech, debate_id=debate, speaker=speaker, party_at_date=party, office_at_date=office,
                  date=day, house=house, section=section, heading=heading, url=URL, rank=chunk,
                  sentences=[C.Sentence(chunk_id=chunk, text=f"Sentence {chunk} on fracking.", matched=True,
                                        column_ref=None)])  # fmt: skip


def deck(*cards):
    return C.Deck(cards=list(cards))


def test_a_lean_deck_states_every_field_once_and_no_url():
    d = deck(card("s1", chunk=1), card("s2", speaker="Michael Shanks", party="Lab", office="Minister for Energy",
                                       chunk=2))
    lean = d.render(lean=True)
    assert URL not in lean and "http" not in lean
    assert lean.count("2026-09-15") == 1 and lean.count("Fracking ===") == 1, "the debate line states them once"
    assert "Commons / Westminster Hall" in lean.splitlines()[0]
    for needle in ("SPEECH s1 | Alison Hume | Lab", "SPEECH s2 | Michael Shanks | Lab | Minister for Energy",
                   "Sentence 1 on fracking.", "Sentence 2 on fracking."):
        assert needle in lean
    assert C.estimate_tokens(lean) < C.estimate_tokens(d.render())


def test_a_card_that_differs_from_its_debate_line_keeps_its_own_line():
    """Lossless by construction: a field is dropped from a card only where the debate line already states it."""
    d = deck(card("s1", chunk=1), card("s2", day=date(2026, 9, 16), section="Chamber", chunk=2))
    lean = d.render(lean=True)
    assert "  2026-09-16 | Commons / Chamber | debate: Fracking" in lean
    assert lean.count("2026-09-15") == 1


def test_the_default_render_is_unchanged():
    d = deck(card("s1", chunk=1))
    assert URL in d.render() and "  2026-09-15 | Commons / Westminster Hall | debate: Fracking" in d.render()
    assert d.render() == d.render(lean=False)


def test_the_lean_variant_is_the_adopted_prompt_with_lean_cards_and_its_own_version():
    lean, base = PR.variant("brief_premise_lean"), PR.variant("brief_premise")
    assert lean.lean_cards and not base.lean_cards
    assert lean.system() == base.system() and lean.version != base.version


def test_the_generator_reads_the_deck_its_variant_asks_for():
    seen = []

    class _Client:
        model = "deepseek-flash"

        async def chat(self, prompt, *, system="", max_tokens=0, temperature=0.0, json_object=False):
            seen.append(prompt)

            class R:
                content = json.dumps({"claims": [], "abstention": "not_in_record", "why": "x"})
                model, fingerprint, finish_reason = "deepseek-flash", "fp", "stop"
                prompt_tokens, completion_tokens, cached_tokens, total_tokens = 10, 1, 0, 11

                def cost_usd(self, model=None):
                    return 0.0

            return R()

    d = deck(card("s1", chunk=1))
    run(G.answer_question("What was said about fracking?", d, _Client(), PROV, variant_key="brief_premise"))
    run(G.answer_question("What was said about fracking?", d, _Client(), PROV, variant_key="brief_premise_lean"))
    assert URL in seen[0] and URL not in seen[1]
