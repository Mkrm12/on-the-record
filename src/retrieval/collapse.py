"""Collapse sibling hits before the reranker sees them."""

from dataclasses import dataclass, field
from typing import Generic, Protocol, Sequence, TypeVar

WINDOW_RADIUS = 2  # ingest.chunk.DEFAULT_WINDOW // 2; a test keeps the two in step
OVERLAP_SPAN = 2 * WINDOW_RADIUS


class Hit(Protocol):
    chunk_id: int
    speech_id: str
    sent_idx: int
    parent_body: str


T = TypeVar("T", bound=Hit)


@dataclass
class Cluster(Generic[T]):
    anchor: T  # the best-ranked hit: the one that stands for the whole cluster
    siblings: list[T] = field(default_factory=list)  # lower-ranked hits it absorbed, in rank order

    @property
    def members(self) -> list[T]:
        return [self.anchor, *self.siblings]


def collapse(hits: Sequence[T], *, span: int = OVERLAP_SPAN, limit: int | None = None) -> list[Cluster[T]]:
    """Clusters in the order their anchors ranked, from `hits` given best first. `limit` keeps the first N clusters (it
    counts clusters, not hits).
    """
    clusters: list[Cluster[T]] = []
    by_speech: dict[str, list[Cluster[T]]] = {}
    by_text: dict[str, Cluster[T]] = {}
    seen: set[int] = set()
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        home = next((c for c in by_speech.get(hit.speech_id, ()) if abs(c.anchor.sent_idx - hit.sent_idx) <= span), None)
        if home is None:
            home = by_text.get(hit.parent_body)
        if home is not None:
            home.siblings.append(hit)
            continue
        cluster = Cluster(hit)
        clusters.append(cluster)
        by_speech.setdefault(hit.speech_id, []).append(cluster)
        by_text[hit.parent_body] = cluster
    return clusters if limit is None else clusters[:limit]
