"""fp32 against bf16 on 300 real claim/evidence pairs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verifier import SUPPORT_THRESHOLD, Verifier, VerifierUnavailable  # noqa: E402

# Line-buffered, because `print` block-buffers when stdout is a pipe.
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEED = 20260928

# Candidates built per reported pair.
POOL_MULTIPLIER = 2

# 's gate.
AGREEMENT_GATE = 0.98
# Scores within this of the threshold are "borderline": where a coarser mantissa could flip a verdict.
BORDERLINE = 0.15
# How much of the reported sample is drawn from that band, if enough candidates exist.
BORDERLINE_SHARE = 0.40

TARGET_DB = "hansard_v2"
PROTECTED = frozenset({"hansard"})

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def dsn_for(database: str) -> str:
    """A read-only look at the corpus. Refuses the rollback the way every other write path does, even though this
    script only reads: a DSN pointed at the wrong database is a mistake worth catching early.
    """
    raw = os.environ.get("PG_DSN")
    if not raw:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("PG_DSN"):
                    raw = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not raw:
        raise SystemExit("PG_DSN is not set and no .env entry was found")
    if database in PROTECTED:
        raise SystemExit(f"refusing to read {database!r} for this benchmark; use {TARGET_DB!r}")
    return psycopg.conninfo.make_conninfo(raw, dbname=database)


def usable_sentences(text: str, min_words: int = 8, max_words: int = 60) -> list[str]:
    """Sentences long enough to be a claim and short enough to be one claim."""
    out = []
    for raw in _SENTENCE.split((text or "").replace("\n", " ")):
        s = raw.strip()
        if min_words <= len(s.split()) <= max_words:
            out.append(s)
    return out


def build_candidates(conn, wanted: int, rng: random.Random) -> list[dict]:
    """Candidate pairs from real speeches."""
    rows = conn.execute("""
        SELECT sp.id, sp.debate_id, sp.text
        FROM speech sp
        WHERE NOT sp.is_boilerplate AND length(sp.text) BETWEEN 400 AND 4000
        ORDER BY md5(sp.id)
        LIMIT %s
    """, (wanted * POOL_MULTIPLIER,)).fetchall()

    by_debate: dict[str, list[tuple[str, str]]] = {}
    for speech_id, debate_id, text in rows:
        by_debate.setdefault(debate_id, []).append((speech_id, text))

    pool = [(sid, did, txt) for did, items in by_debate.items() for sid, txt in items]
    candidates: list[dict] = []
    for speech_id, debate_id, text in pool:
        sentences = usable_sentences(text)
        if not sentences:
            continue
        candidates.append({"kind": "self", "speech_id": speech_id, "debate_id": debate_id,
                           "evidence": text, "claim": rng.choice(sentences)})  # fmt: skip

        siblings = [t for sid, t in by_debate[debate_id] if sid != speech_id]
        if siblings:
            sib = usable_sentences(rng.choice(siblings))
            if sib:
                candidates.append({"kind": "sibling", "speech_id": speech_id, "debate_id": debate_id,
                                   "evidence": text, "claim": rng.choice(sib)})  # fmt: skip

        other = rng.choice(pool)
        if other[1] != debate_id:
            far = usable_sentences(other[2])
            if far:
                candidates.append({"kind": "unrelated", "speech_id": speech_id, "debate_id": debate_id,
                                   "evidence": text, "claim": rng.choice(far)})  # fmt: skip
    rng.shuffle(candidates)
    return candidates


def stratify(candidates: list[dict], scores: np.ndarray, wanted: int, rng: random.Random) -> list[int]:
    """Indices of the pairs to report: a fixed share borderline, the rest spread over the score range."""
    borderline = [i for i, s in enumerate(scores) if abs(float(s) - SUPPORT_THRESHOLD) <= BORDERLINE]
    rest = [i for i in range(len(scores)) if i not in set(borderline)]
    rng.shuffle(borderline)
    rng.shuffle(rest)
    take_border = min(len(borderline), int(wanted * BORDERLINE_SHARE))
    chosen = borderline[:take_border]
    chosen += rest[:wanted - len(chosen)]
    # If the corpus simply has few borderline pairs, say so rather than pretending the sample is balanced.
    return sorted(chosen[:wanted])


def agreement(a: np.ndarray, b: np.ndarray, threshold: float = SUPPORT_THRESHOLD) -> float:
    if not len(a):
        return float("nan")
    return float(((a >= threshold) == (b >= threshold)).mean())


def wilson_lower(successes: int, n: int, z: float = 1.96) -> float:
    """A 95% LOWER bound on the agreement rate. The gate is a floor, so the bound that matters is the low one: 295 of
    300 is 98.3%, and its lower bound is 96.2%, which does not clear a 98% floor.
    """
    if n == 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    spread = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return max(0.0, (centre - spread) / (1 + z * z / n))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--database", default=TARGET_DB)
    p.add_argument("--batch-size", type=int, default=8,
               help="fp32 T5-large at 16 x long evidence is the likeliest cause of the first "
                    "run being killed; 8 halves the peak activation footprint")  # fmt: skip
    p.add_argument("--out", default="e2_verifier_precision.json")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    rng = random.Random(SEED)
    started = time.time()
    with psycopg.connect(dsn_for(args.database)) as conn:
        candidates = build_candidates(conn, args.pairs, rng)
    if len(candidates) < args.pairs:
        raise SystemExit(f"only {len(candidates)} candidate pairs could be built; need {args.pairs}")
    print(f"{len(candidates)} candidate pairs built from {args.database}")

    # --- fp32: the reference ---------------------------------------------------------------------------
    fp32 = Verifier(precision="fp32", batch_size=args.batch_size)
    print("fp32 self-check:", fp32.self_check())
    t0 = time.perf_counter()
    chunk_size = 100
    parts: list[np.ndarray] = []
    for start in range(0, len(candidates), chunk_size):
        slice_ = candidates[start:start + chunk_size]
        parts.append(np.array(fp32.score([c["evidence"] for c in slice_], [c["claim"] for c in slice_])))
        done = min(start + chunk_size, len(candidates))
        rate = (time.perf_counter() - t0) / done
        print(f"  fp32 pool {done}/{len(candidates)}  {rate:.2f}s/pair  "
              f"eta {rate * (len(candidates) - done) / 60:.1f} min")
    pool_scores = np.concatenate(parts)
    pool_seconds = time.perf_counter() - t0
    print(f"fp32 scored the {len(candidates)}-pair pool in {pool_seconds:.1f}s")

    chosen = stratify(candidates, pool_scores, args.pairs, rng)
    pairs = [candidates[i] for i in chosen]
    fp32_scores = pool_scores[chosen]

    # Rescore the chosen pairs alone, and use THAT as the reference.
    pool_reference = fp32_scores
    fp32.score([pairs[0]["evidence"]], [pairs[0]["claim"]])  # warm the kernels before timing
    t0 = time.perf_counter()
    fp32_scores = np.array(fp32.score([c["evidence"] for c in pairs], [c["claim"] for c in pairs]))
    fp32_seconds = time.perf_counter() - t0
    borderline_mask = np.abs(fp32_scores - SUPPORT_THRESHOLD) <= BORDERLINE
    # How much the batching alone moves a score, at one fixed precision.
    batching_diff = np.abs(pool_reference - fp32_scores)
    batching_agreement = agreement(pool_reference, fp32_scores)
    fp32.close()
    del fp32
    import torch
    torch.cuda.empty_cache()

    # --- bf16: the candidate --------------------------------------------------------------------------
    bf16 = Verifier(precision="bf16", batch_size=args.batch_size)
    print("bf16 self-check:", bf16.self_check())
    bf16.score([pairs[0]["evidence"]], [pairs[0]["claim"]])  # warm, so the timing below compares like with like
    t0 = time.perf_counter()
    bf16_scores = np.array(bf16.score([c["evidence"] for c in pairs], [c["claim"] for c in pairs]))
    bf16_seconds = time.perf_counter() - t0
    bf16.close()

    same = (fp32_scores >= SUPPORT_THRESHOLD) == (bf16_scores >= SUPPORT_THRESHOLD)
    agreed = int(same.sum())
    n = len(pairs)
    overall = agreed / n
    lower = wilson_lower(agreed, n)
    border_n = int(borderline_mask.sum())
    border_agreement = agreement(fp32_scores[borderline_mask], bf16_scores[borderline_mask])
    diffs = np.abs(fp32_scores - bf16_scores)

    # The gate is read against the LOWER bound, and a borderline band that disagrees fails on its own.
    adopt = bool(lower >= AGREEMENT_GATE and (border_n == 0 or border_agreement >= AGREEMENT_GATE))
    decision = "bf16" if adopt else "fp32"

    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "database": args.database, "seed": SEED, "pairs": n,
        "candidate_pool": len(candidates),
        "threshold": SUPPORT_THRESHOLD,
        "gate": AGREEMENT_GATE,
        "kinds": {k: sum(1 for c in pairs if c["kind"] == k) for k in ("self", "sibling", "unrelated")},
        # Scoring is batch-sensitive, so the fp32 reference is scored with the same batching as bf16.
        "fp32_batching_agreement": round(batching_agreement, 4),
        "fp32_batching_max_abs_diff": round(float(batching_diff.max()), 6),
        "fp32_batching_mean_abs_diff": round(float(batching_diff.mean()), 6),
        "agreement": round(overall, 4),
        "agreed": agreed,
        "disagreed": n - agreed,
        "agreement_wilson_lower_95": round(lower, 4),
        "borderline_band": round(BORDERLINE, 3),
        "borderline_pairs": border_n,
        "borderline_agreement": None if border_n == 0 else round(border_agreement, 4),
        "max_abs_score_diff": round(float(diffs.max()), 6),
        "mean_abs_score_diff": round(float(diffs.mean()), 6),
        "p99_abs_score_diff": round(float(np.percentile(diffs, 99)), 6),
        "pearson_r": round(float(np.corrcoef(fp32_scores, bf16_scores)[0, 1]), 6),
        "fp32_seconds": round(fp32_seconds, 2),
        "bf16_seconds": round(bf16_seconds, 2),
        "bf16_speedup": round(fp32_seconds / bf16_seconds, 3) if bf16_seconds else None,
        "fp32_median_score": round(float(statistics.median(fp32_scores)), 4),
        "decision": decision,
        "adopt_bf16": adopt,
        "reason": (
            f"bf16 agrees with fp32 on {agreed}/{n} pairs ({overall:.1%}); the 95% lower bound is {lower:.1%} "
            f"and the borderline band agrees "
            f"{'n/a' if border_n == 0 else f'{border_agreement:.1%} over {border_n} pairs'}. "
            f"{'Adopted' if adopt else 'Not adopted'} against a {AGREEMENT_GATE:.0%} floor."
        ),
        "disagreements": [
            {"kind": pairs[i]["kind"], "speech_id": pairs[i]["speech_id"],
             "fp32": round(float(fp32_scores[i]), 6), "bf16": round(float(bf16_scores[i]), 6),
             "claim": pairs[i]["claim"][:160]}  # fmt: skip
            for i in np.where(~same)[0][:20]
        ],
        "total_seconds": round(time.time() - started, 1),
    }

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / args.out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\npairs                 {n}  ({report['kinds']})")
        print(f"fp32 batch sensitivity {report['fp32_batching_agreement']:.2%} same verdict, "
              f"max score move {report['fp32_batching_max_abs_diff']:.6f} (same precision, different batching)")
        print(f"agreement             {agreed}/{n} = {overall:.2%}  (95% lower bound {lower:.2%})")
        print(f"borderline band       {border_n} pairs within {BORDERLINE} of {SUPPORT_THRESHOLD}"
              f"{'' if border_n == 0 else f', agreement {border_agreement:.2%}'}")
        print(f"score difference      mean {report['mean_abs_score_diff']:.6f}  "
              f"p99 {report['p99_abs_score_diff']:.6f}  max {report['max_abs_score_diff']:.6f}")
        print(f"pearson r             {report['pearson_r']:.6f}")
        print(f"speed                 fp32 {fp32_seconds:.1f}s vs bf16 {bf16_seconds:.1f}s "
              f"({report['bf16_speedup']}x)")
        print(f"\nDECISION: run the verifier in {decision}.")
        print(report["reason"])
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VerifierUnavailable as exc:
        print(f"verifier unavailable: {exc}")
        sys.exit(2)
