"""MiniCheck-Flan-T5-Large: does this evidence support this claim?"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("gpu.verifier")

# Where the weights are.
DEFAULT_PATH = os.environ.get("MINICHECK_PATH", str(Path.home() / "models" / "MiniCheck-Flan-T5-Large"))

VERIFIER_NAME = "lytang/MiniCheck-Flan-T5-Large"
VERIFIER_REVISION = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"

# The two vocabulary positions that are the classifier.
NO_SUPPORT_TOKEN = 3
SUPPORT_TOKEN = 209

# The model card's own values.
MAX_INPUT_TOKENS = 2048
CHUNK_WORDS = 500
DEFAULT_BATCH_SIZE = 16

# The precision the benchmark adopted, and the default every caller gets. fp32 remains available and is what a re-
# measurement compares against; changing this constant is a decision that needs its own 300-pair run.
ADOPTED_PRECISION = "bf16"

# Above this, supported. gives no threshold, so 0.5 — the point at which the two-way softmax changes its mind — is the
# only one that is not an invention.
SUPPORT_THRESHOLD = 0.5

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class VerifierUnavailable(RuntimeError):
    """The model could not be loaded. Callers degrade rather than fail: allows a verbatim `said` claim to pass on the
    deterministic checks alone, and says a paraphrase must be dropped.
    """


def sentences(text: str) -> list[str]:
    """Split on sentence boundaries, keeping paragraph breaks."""
    parts: list[str] = []
    for block in text.split("\n"):
        parts.extend(s for s in _SENTENCE.split(block.strip()) if s)
    return parts or [""]


def chunk(text: str, words: int = CHUNK_WORDS) -> list[str]:
    """Group sentences into chunks of about `words` words, as the reference does."""
    out: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences(text):
        n = len(sentence.split())
        if current and count + n > words:
            out.append(" ".join(current))
            current, count = [sentence], n
        else:
            current.append(sentence)
            count += n
    if current:
        out.append(" ".join(current))
    return [c.strip() for c in out if c.strip()] or [""]


class Verifier:
    """MiniCheck on one GPU. Scores (evidence, claim) pairs; higher means better supported."""

    def __init__(self, path: str = DEFAULT_PATH, device: str = "cuda:0", precision: str = ADOPTED_PRECISION,
                 batch_size: int = DEFAULT_BATCH_SIZE):  # fmt: skip
        if precision not in ("fp32", "bf16"):
            # fp16 is refused outright rather than offered and warned about.
            raise VerifierUnavailable(
                f"precision must be fp32 or bf16, not {precision!r}. Flan-T5 overflows in fp16 and emits NaNs, "
                "which arrive downstream as a confident-looking 0.5.")
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment, not logic
            raise VerifierUnavailable(f"torch/transformers unavailable: {exc}") from exc

        if not Path(path).exists():
            raise VerifierUnavailable(f"no MiniCheck weights at {path}")

        self._torch = torch
        self.path, self.device, self.precision = path, device, precision
        self.batch_size = batch_size
        dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=dtype).to(device)
        except Exception as exc:
            raise VerifierUnavailable(f"could not load MiniCheck from {path}: {exc}") from exc
        self.model.eval()
        self._check_label_tokens()

    def _check_label_tokens(self) -> None:
        """The two label positions are the whole classifier, so they are derived and checked, never assumed."""
        first_no = self.tokenizer("0").input_ids[0]
        first_yes = self.tokenizer("1").input_ids[0]
        if (first_no, first_yes) != (NO_SUPPORT_TOKEN, SUPPORT_TOKEN):
            raise VerifierUnavailable(
                f"this tokenizer starts the label strings at ids {first_no}/{first_yes}, but MiniCheck reads its "
                f"verdict off logits {NO_SUPPORT_TOKEN}/{SUPPORT_TOKEN}. Against this vocabulary those are two "
                "arbitrary logits, and every score would be meaningless while still looking like a probability.")
        log.info("MiniCheck label tokens verified: '0'->%d %r, '1'->%d %r", NO_SUPPORT_TOKEN,
                 self.tokenizer.convert_ids_to_tokens(NO_SUPPORT_TOKEN), SUPPORT_TOKEN,
                 self.tokenizer.convert_ids_to_tokens(SUPPORT_TOKEN))  # fmt: skip

    # Two pairs with no room for argument, used to prove the wiring end to end at load time.
    SANITY_PAIRS = (
        ("The Minister said funding would rise to 35 billion pounds next year.",
         "The Minister said funding would rise to 35 billion pounds.", True),
        ("The Minister said funding would rise to 35 billion pounds next year.",
         "The Minister said funding would be cut by half.", False),
    )

    def self_check(self) -> dict:
        """Score two unambiguous pairs and confirm the supported one scores higher."""
        evidence = [p[0] for p in self.SANITY_PAIRS]
        claims = [p[1] for p in self.SANITY_PAIRS]
        scores = self.score(evidence, claims)
        supported, contradicted = scores[0], scores[1]
        if not supported > contradicted:
            raise VerifierUnavailable(
                f"MiniCheck scored a supported claim {supported:.4f} and a contradicted one "
                f"{contradicted:.4f}. The label logits are right, so the pipeline is wired the wrong way round "
                "— check the prompt order and the softmax column.")
        return {"supported": round(supported, 4), "contradicted": round(contradicted, 4),
                "separation": round(supported - contradicted, 4)}  # fmt: skip

    @property
    def dtype(self) -> str:
        return self.precision

    def _score_pairs(self, docs: list[str], claims: list[str]) -> np.ndarray:
        """Support probability for each (doc, claim) pair, in order."""
        torch = self._torch
        eos = self.tokenizer.eos_token or "</s>"
        texts = [f"predict: {doc}{eos}{claim}" for doc, claim in zip(docs, claims, strict=True)]
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self.tokenizer(batch, max_length=MAX_INPUT_TOKENS, truncation=True, padding=True,
                                     return_tensors="pt")  # fmt: skip
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                decoder_input_ids = torch.zeros((encoded["input_ids"].size(0), 1), dtype=torch.long,
                                                device=self.device)  # fmt: skip
                logits = self.model(input_ids=encoded["input_ids"],
                                    attention_mask=encoded["attention_mask"],
                                    decoder_input_ids=decoder_input_ids).logits.squeeze(1)  # fmt: skip
                pair = logits[:, torch.tensor([NO_SUPPORT_TOKEN, SUPPORT_TOKEN], device=logits.device)]
                # float before softmax: under bf16 the two logits can be within one mantissa step of each other, and
                # softmaxing in bf16 then rounds the probability to a coarser grid than the decision needs.
                probs = torch.nn.functional.softmax(pair.float(), dim=-1)
            out.append(probs[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def score(self, evidence: list[str], claims: list[str]) -> list[float]:
        """One support probability per claim, aggregating by max over the evidence's chunks."""
        if len(evidence) != len(claims):
            raise ValueError("evidence and claims must be the same length")
        flat_docs: list[str] = []
        flat_claims: list[str] = []
        spans: list[tuple[int, int]] = []
        for doc, claim in zip(evidence, claims, strict=True):
            pieces = chunk(doc)
            spans.append((len(flat_docs), len(flat_docs) + len(pieces)))
            flat_docs.extend(pieces)
            flat_claims.extend([claim] * len(pieces))
        scores = self._score_pairs(flat_docs, flat_claims)
        return [float(scores[a:b].max()) if b > a else 0.0 for a, b in spans]

    def verdicts(self, evidence: list[str], claims: list[str],
                 threshold: float = SUPPORT_THRESHOLD) -> list[dict]:  # fmt: skip
        return [
            {"score": s, "verdict": "supported" if s >= threshold else "unsupported", "threshold": threshold}
            for s in self.score(evidence, claims)
        ]

    def info(self) -> dict:
        return {"name": VERIFIER_NAME, "revision": VERIFIER_REVISION, "precision": self.precision,
                "threshold": SUPPORT_THRESHOLD, "max_input_tokens": MAX_INPUT_TOKENS,
                "label_tokens": [NO_SUPPORT_TOKEN, SUPPORT_TOKEN]}  # fmt: skip

    def close(self) -> None:
        self.model = self.tokenizer = None
