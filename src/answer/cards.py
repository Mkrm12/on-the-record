"""Evidence cards: the only thing the generator is allowed to read."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any

# The default context budget for cards alone, before the prompt and the question. says about 8k.
DEFAULT_TOKEN_BUDGET = 8_000

# Characters per token for English prose.
CHARS_PER_TOKEN = 4.0

# How many sentences either side of a match a card shows. says ±2.
CONTEXT_SENTENCES = 2


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass
class Sentence:
    """One sentence of the record, with the id a claim must cite to use it."""

    chunk_id: int
    text: str
    matched: bool = False
    column_ref: str | None = None

    @property
    def citable(self) -> bool:
        """Whether a claim may cite this sentence by id: whenever its id is known."""
        return self.chunk_id >= 0

    def render(self) -> str:
        """The sentence, with a citable id only where there is one."""
        if not self.citable:
            return f"  (context, not citable) {self.text}"
        mark = "*" if self.matched else " "
        return f"  [{self.chunk_id}]{mark} {self.text}"


@dataclass
class Card:
    """One speech: who said it, when, where, and the sentences themselves."""

    speech_id: str
    debate_id: str
    sentences: list[Sentence] = field(default_factory=list)
    speaker: str | None = None
    party_at_date: str | None = None
    office_at_date: str | None = None
    date: Date | None = None
    house: str | None = None
    section: str | None = None
    heading: str | None = None
    url: str | None = None
    rank: int = 0
    score: float = 0.0
    # The cross-encoder's ABSOLUTE relevance score, kept separately from `score` — which under `rerank: "blend"` is
    # dominated by the fused RRF and therefore measures arm agreement rather than relevance (AUC 0.562 against 0.691 for
    # this one, on 108 dev items).
    rerank_score: float | None = None

    @property
    def chunk_ids(self) -> tuple[int, ...]:
        return tuple(s.chunk_id for s in self.sentences)

    @property
    def matched_chunk_ids(self) -> tuple[int, ...]:
        return tuple(s.chunk_id for s in self.sentences if s.matched)

    @property
    def text(self) -> str:
        """The card's evidence as one string — what the verifier's quote check compares against."""
        return " ".join(s.text for s in self.sentences)

    def header(self) -> str:
        """Everything needed to attribute a claim, and nothing that invites one."""
        who = self.speaker or "an unnamed speaker"
        bits = [who]
        if self.party_at_date:
            bits.append(self.party_at_date)
        if self.office_at_date:
            bits.append(self.office_at_date)
        where = " / ".join(x for x in (self.house, self.section) if x)
        when = self.date.isoformat() if self.date else "date not recorded"
        return (f"SPEECH {self.speech_id} | {' | '.join(bits)}\n"
                f"  {when} | {where} | debate: {self.heading or 'untitled'}"
                + (f"\n  {self.url}" if self.url else ""))  # fmt: skip

    def render(self) -> str:
        return self.header() + "\n" + "\n".join(s.render() for s in self.sentences)

    def place(self) -> tuple:
        """What a debate line states for its cards: date, House, section, heading."""
        return (self.date, self.house, self.section, self.heading)

    def lean_header(self, debate_place: tuple) -> str:
        """Who spoke — speaker, party, office — always; date / House / section / heading only where they differ from
        the debate line above (on dev: never, 0 of 865 cards); no URL (citations take it from the card's data).
        """
        bits = [self.speaker or "an unnamed speaker"] + [x for x in (self.party_at_date, self.office_at_date) if x]
        head = f"SPEECH {self.speech_id} | {' | '.join(bits)}"
        if self.place() == debate_place:
            return head
        where = " / ".join(x for x in (self.house, self.section) if x)
        when = self.date.isoformat() if self.date else "date not recorded"
        return f"{head}\n  {when} | {where} | debate: {self.heading or 'untitled'}"

    def tokens(self) -> int:
        return estimate_tokens(self.render())


def card_from_hit(hit: Any, speaker: str | None = None) -> Card:
    """One card from one retrieval hit, using the context window retrieval already expanded."""
    matched = {hit.chunk_id, *(hit.sibling_chunk_ids or ())}
    sentences: list[Sentence] = []
    if hit.context_sentences:
        # (seq, sent_idx, body, column_ref, chunk_id).
        entries = list(hit.context_sentences)
        # Trimmed to ±CONTEXT_SENTENCES here rather than by reconfiguring retrieval.
        anchor = next((i for i, e in enumerate(entries) if (e[0], e[1]) == (hit.seq, hit.sent_idx)), None)
        if anchor is not None:
            low = max(0, anchor - CONTEXT_SENTENCES)
            entries = entries[low:anchor + CONTEXT_SENTENCES + 1]
        for entry in entries:
            seq, sent_idx, body, column_ref = entry[0], entry[1], entry[2], entry[3]
            given = entry[4] if len(entry) > 4 else None
            if (seq, sent_idx) == (hit.seq, hit.sent_idx):
                chunk_id = hit.chunk_id
            else:
                chunk_id = given if isinstance(given, int) and given >= 0 else -1
            sentences.append(Sentence(chunk_id=chunk_id, text=body, matched=chunk_id in matched,
                                      column_ref=column_ref))  # fmt: skip
    if not sentences:
        sentences = [Sentence(chunk_id=hit.chunk_id, text=hit.sentence, matched=True,
                              column_ref=hit.column_ref)]  # fmt: skip
    return Card(
        speech_id=hit.speech_id, debate_id=hit.debate_id, sentences=sentences,
        speaker=speaker or hit.speaker, party_at_date=hit.party, office_at_date=hit.office,
        date=hit.date, house=hit.house, section=getattr(hit, "section", None),
        heading=hit.heading, url=hit.url, rank=hit.rank, score=hit.score, rerank_score=hit.rerank_score,
    )  # fmt: skip


@dataclass
class Deck:
    """The cards that fit the budget, and an honest account of what did not."""

    cards: list[Card] = field(default_factory=list)
    dropped: int = 0
    budget_tokens: int = DEFAULT_TOKEN_BUDGET
    used_tokens: int = 0

    @property
    def chunk_ids(self) -> frozenset[int]:
        """Every sentence id the generator may cite. Anything else is a fabricated citation, and the contract refuses
        to build a claim that cites outside this set.
        """
        return frozenset(c for card in self.cards for c in card.chunk_ids if c >= 0)

    @property
    def debate_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for card in self.cards:
            if card.debate_id not in seen:
                seen.append(card.debate_id)
        return tuple(seen)

    @property
    def speech_ids(self) -> tuple[str, ...]:
        return tuple(card.speech_id for card in self.cards)

    def render(self, lean: bool = False) -> str:
        """Cards grouped by debate and ordered by date, which is how a reader would read them."""
        blocks: list[str] = []
        for debate_id in self.debate_ids:
            group = [c for c in self.cards if c.debate_id == debate_id]
            head = group[0]
            when = head.date.isoformat() if head.date else "date not recorded"
            if lean:
                where = " / ".join(x for x in (head.house, head.section) if x) or "?"
                blocks.append(f"=== DEBATE {debate_id} | {when} | {where} | {head.heading or 'untitled'} ===")
                blocks.extend(card.lean_header(head.place()) + "\n" + "\n".join(s.render() for s in card.sentences)
                              for card in group)  # fmt: skip
                continue
            blocks.append(f"=== DEBATE {debate_id} | {when} | {head.house or '?'} | "
                          f"{head.heading or 'untitled'} ===")  # fmt: skip
            blocks.extend(card.render() for card in group)
        return "\n\n".join(blocks)


def _sort_key(card: Card) -> tuple:
    return (card.date or Date.min, card.debate_id, card.rank)


def build_deck(hits: list[Any], budget_tokens: int = DEFAULT_TOKEN_BUDGET,
               speakers: Mapping[str, str] | None = None) -> Deck:  # fmt: skip
    """Cards for the hits that fit, best first, then ordered for reading."""
    deck = Deck(budget_tokens=budget_tokens)
    named = speakers or {}
    by_rank = sorted((card_from_hit(h, named.get(h.speech_id)) for h in hits), key=lambda c: c.rank)
    kept: list[Card] = []
    used = 0
    for card in by_rank:
        cost = card.tokens()
        if used + cost > budget_tokens and kept:
            deck.dropped += 1
            continue
        # The first card goes in even if it alone exceeds the budget: a deck with no evidence cannot produce an answer,
        # and the sufficiency check should be the thing that declines, not an arithmetic accident.
        kept.append(card)
        used += cost
    deck.cards = sorted(kept, key=_sort_key)
    deck.used_tokens = used
    return deck


def render_structured_facts(facts: list[dict[str, Any]] | None) -> str:
    """Facts from SQL, labelled so the model cannot mistake them for something to derive."""
    if not facts:
        return ""
    lines = ["=== FACTS FROM THE DATABASE (already computed; use them as given, do not recalculate) ==="]
    for fact in facts:
        value = fact.get("value")
        body = ", ".join(f"{k}: {v}" for k, v in (value or {}).items() if v is not None) or "not recorded"
        lines.append(f"- {fact.get('template', 'fact')}: {body}")
        if fact.get("caveat"):
            lines.append(f"  CAVEAT (must be repeated in the answer if this fact is used): {fact['caveat']}")
    return "\n".join(lines)
