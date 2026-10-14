# Session 3 final report: the adopted recipe, measured on the production corpus

Scored 2026-09-26T03:22:02 against `hansard_v2`.

**Recipe:** `BAAI/bge-small-en-v1.5|full|v2` -- the matched sentence under a
`[date | house | heading | speaker | party]` header, embedded with BAAI/bge-small-en-v1.5 (384
dimensions). All 1,074,297 embeddable chunks carry this one recipe; the
HNSW index (m=16, ef_construction=64) was rebuilt over the finished table and the table vacuumed.

**Search configuration: unchanged.** The Session 3 knob sweep adopted nothing, because the
baseline was already at a plateau on every knob. So every difference below is the embedding
recipe and nothing else.

**Relevance:** the judged pool binarised at UMBRELA grade >= 2. The judge
passes its calibration gate on that one bit and fails it on the four-level grade (I-77), so nDCG
here is a binarised nDCG and no graded number is claimed.

## Headline

| split | queries | recall@100 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| dev (W0, W1, W2, W4) | 108 | 0.6379 | 0.5681 | 0.4946 | 0.4557 |
| test | 93 | 0.5258 | 0.4720 | 0.4552 | 0.4016 |
| held-out W3 | 36 | 0.5559 | 0.4953 | 0.5299 | 0.4320 |

### Against the pre-adoption baseline

| split | recall@100 | recall@10 | MRR@10 | nDCG@10 | before, from |
|---|---|---|---|---|---|
| dev (W0, W1, W2, W4) | +0.0784 | +0.0612 | +0.0269 | +0.0317 | `data/tuning/sweep_dev.json` |
| test | +0.0113 | +0.0048 | +0.0481 | +0.0351 | `reports/session3_confirm.json` |
| held-out W3 | +0.0672 | +0.0522 | +0.1243 | +0.0671 | `reports/session3_final_exam.json` |

On dev the before/after comparison is **paired**: the same 108 queries, the same
judged relevance, the old rankings read back from the sweep cache. Mean difference in
recall@100 **+0.0784**, 95% CI [+0.0054, +0.1526], p = 0.0384 -- established after
the family correction.

**Dev is not independent evidence for the choice, and this report does not treat it as such.**
The bake-off screened the header recipes on these same dev queries, so dev is where the recipe
was selected. What the dev number establishes is that the gain *transfers*: from a 120,000-chunk
comparative corpus to 1.07M chunks, and from dense-only exact search to the full three-leg
pipeline with the reranker. Those are the two things that could have made it evaporate. The
out-of-sample evidence is the test split, which had no part in the choice.

On test and W3 only the aggregate metrics were written down before adoption, so those rows are
a difference between two numbers and not a paired test. They are reported as such.

## The first-stage ceiling

What the reranker is handed. Nothing it does can recover a relevant speech the first stage
never retrieved, so this bounds the whole pipeline: it is the number to attack next.

| split | first-stage recall@100 | pipeline recall@10 | headroom the reranker cannot reach |
|---|---|---|---|
| dev (W0, W1, W2, W4) | 0.7444 | 0.5681 | 0.2556 |
| test | 0.6520 | 0.4720 | 0.3480 |
| held-out W3 | 0.7189 | 0.4953 | 0.2811 |

## By question category (recall@10)

The average is not the deliverable. A recipe that lifts the mean while losing a question type
is not an upgrade, and the only way to see that is to look category by category. Sorted by how
many queries the category carries, because the rows at the bottom are two or three questions
and cannot support a conclusion on their own.

| category | queries | dev (W0, W1, W2, W4) | test | held-out W3 | dev change |
|---|---|---|---|---|---|
| multi_speech | 45 | 0.2321 (n=20) | 0.1951 (n=17) | 0.2306 (n=8) | +0.0580 |
| who_spoke | 40 | 0.6250 (n=16) | 0.2917 (n=16) | 0.1250 (n=8) | +0.3125 |
| paraphrase | 17 | 0.3864 (n=11) | 0.6667 (n=3) | 0.5128 (n=3) | +0.0909 |
| exact_term | 13 | 0.9500 (n=8) | 0.8750 (n=4) | 0.5000 (n=1) | +0.1250 |
| colloquial | 12 | 0.4381 (n=5) | 0.3917 (n=6) | 0.0000 (n=1) | -0.0476 |
| named_speaker | 11 | 0.6667 (n=6) | 0.9375 (n=4) | 1.0000 (n=1) | 0.0000 |
| numeric | 11 | 0.7981 (n=6) | 1.0000 (n=3) | 1.0000 (n=2) | +0.0037 |
| false_premise | 8 | 0.8333 (n=3) | 0.5000 (n=3) | 1.0000 (n=2) | +0.3333 |
| loaded | 6 | 0.9000 (n=2) | 0.2500 (n=4) | - | +0.1500 |
| misattribution | 6 | 0.5000 (n=2) | 0.3333 (n=3) | 1.0000 (n=1) | 0.0000 |
| numeric_perturbation | 6 | 1.0000 (n=2) | 0.7500 (n=2) | 0.6667 (n=2) | 0.0000 |
| quote_verification | 6 | 1.0000 (n=2) | 1.0000 (n=3) | 1.0000 (n=1) | +0.5000 |
| scoping | 6 | 0.6127 (n=3) | 0.7500 (n=2) | 0.7143 (n=1) | -0.1619 |
| temporal | 6 | 0.8571 (n=2) | 0.5000 (n=2) | 0.7000 (n=2) | -0.1429 |
| comparison | 5 | 0.5000 (n=3) | 0.1000 (n=2) | - | 0.0000 |
| cross_window | 5 | 0.1667 (n=2) | 0.2000 (n=3) | - | -0.3000 |
| multi_hop | 5 | 1.0000 (n=1) | 0.6667 (n=3) | 1.0000 (n=1) | 0.0000 |
| multi_part | 5 | 0.3500 (n=2) | 0.3750 (n=2) | 0.5000 (n=1) | -0.3833 |
| negation_flip | 5 | 0.6667 (n=3) | 0.5000 (n=2) | - | -0.3333 |
| office_at_date | 5 | 0.2500 (n=2) | 0.6667 (n=3) | - | -0.2500 |
| party_attribution | 5 | 0.6667 (n=2) | 0.8000 (n=2) | 1.0000 (n=1) | 0.0000 |
| stance | 5 | 0.7222 (n=3) | 0.3333 (n=2) | - | 0.0000 |
| mention_vs_speaker | 4 | 0.7500 (n=2) | 1.0000 (n=2) | - | 0.0000 |

### Did the recipe cost anything?

**Yes, and it is recorded rather than averaged away.** These categories lost more than 0.02
recall@10 on dev against the old recipe:

| category | queries | before | after | change |
|---|---|---|---|---|
| multi_part | 2 | 0.7333 | 0.3500 | -0.3833 |
| negation_flip | 3 | 1.0000 | 0.6667 | -0.3333 |
| cross_window | 2 | 0.4667 | 0.1667 | -0.3000 |
| office_at_date | 2 | 0.5000 | 0.2500 | -0.2500 |
| scoping | 3 | 0.7746 | 0.6127 | -0.1619 |
| temporal | 2 | 1.0000 | 0.8571 | -0.1429 |
| colloquial | 5 | 0.4857 | 0.4381 | -0.0476 |

The bake-off refused qwen3-0.6b for exactly this -- a better mean bought by losing a question
type -- so the rule has to be applied to the recipe that was adopted too. It is applied here as
disclosure and not as a veto, for one reason: the header recipe's gain is broad rather than
concentrated, and every regressed category above rests on a handful of dev queries. Watch them
as the gold set grows; a loss that survives a larger sample is a reason to revisit the recipe.

## Disclosure: W3 has been scored twice

**What happened.** W3 is the held-out window and the discipline is that it is scored once, at
the very end. It was scored on 2026-09-25 (`reports/session3_final_exam.json`) at the close of
the knob sweep -- and then the embedder bake-off changed the embedding recipe, which is a far
larger change than any knob. The exam was sat before the last change to the thing it examines.
That is a sequencing error on my part, not a judgement call: the bake-off should have run
before the final exam.

**What was done about it.** W3 was re-scored under the adopted recipe, on the owner's explicit
instruction, and both numbers are published:

| | recall@100 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| first look (old recipe), 2026-09-25T18:57:10 | 0.4886 | 0.4431 | 0.4056 | 0.3649 |
| second look (adopted recipe), 2026-09-26T03:22:02 | 0.5559 | 0.4953 | 0.5299 | 0.4320 |

**How to read the second number.** It is an honest measurement of the adopted recipe on W3 and
it is *not* a clean held-out result. W3's queries have now informed no tuning decision -- no
knob was adopted and the recipe was chosen on dev alone -- so the number is not contaminated by
selection on W3. What it has lost is the statistical guarantee that comes from a set nobody has
looked at: the strength of the claim it can support is weaker than the first look's, and if a
future change needs a genuinely untouched set, W3 is no longer it. Phase 3 should cut a fresh
held-out window from the corpus and score it once, last.

### Where the value is actually lost

The categories that carry weight (five queries or more across all splits) and still score under
0.45 recall@10, pooled. These, not the headline average, are what Phase 3 has to move:

| category | queries | recall@10 | what is going wrong |
|---|---|---|---|
| cross_window | 5 | 0.1867 | the two halves of the answer sit in different windows, years apart, and nothing in the query tells the dense leg to look in both |
| multi_speech | 45 | 0.2178 | the answer is spread over several contributions and each sentence holds only part of it, so no single chunk looks relevant enough to rank |
| comparison | 5 | 0.3400 | needs both sides to be retrieved, and one side is usually enough to fill ten slots |
| colloquial | 12 | 0.3784 | the query is not in the register of the record, so neither leg has a strong signal |
| multi_part | 5 | 0.3900 | two questions in one; the retrieval serves whichever part dominates the embedding |
| who_spoke | 40 | 0.3917 | a roll-call of participants is a set, not a passage: recall@10 asks for ten speeches when the answer is every speech in a debate |

Two distinct shapes, and they want different fixes:

1. **The answer is a set, and both the retrieval and the metric assume it is one passage.**
   `multi_speech` and `who_spoke` together are 85 of the 237 retrieval-testable questions -- the
   largest block in the gold set, and the weakest. Asking for ten speeches when the answer is
   every contribution to a debate measures the wrong thing; a heading-scoped leg (I-38) that can
   return a whole debate, or a depth chosen per category, attacks it directly.
2. **The query is not in the record's register.** `colloquial` gives the keyword leg nothing to
   match, so it rests entirely on the dense leg; `paraphrase` sits just above the line at 0.46 for
   the same reason. This is what a better embedder or a query rewrite (HyDE) is for -- and the
   model bake-off found the model barely mattered next to the text it was given, so a rewrite is
   the more promising of the two.

Neither is a knob, which is consistent with the sweep finding the knobs already at a plateau.

## Verdict on Phase 2

**Phase 2 is complete. Go for Phase 3.**

What is locked in and reproducible:

* a 1,074,297-chunk corpus embedded under one recipe, with the HNSW
  index rebuilt over it and the table vacuumed and analysed;
* a 488-item gold set, triple-graded, with a judge calibrated against blind expert grades
  (binarised kappa 0.6897, passing; four-level 0.5316, failing, which is why no graded nDCG is
  claimed anywhere in this project);
* an adoption decision taken by a rule written before the numbers were seen, which refused a
  candidate that improved the average and lost a question category;
* retrieval knobs swept, held to a plateau rather than a peak, and nothing adopted on noise.

What Phase 3 inherits as its main problem:

* **the first stage never retrieves 25.6% of relevant speeches at depth 100 on dev**, and no reranker can recover what it was not handed. The categories above say where that loss sits: questions whose answer is a set of passages.
* **the reranker pool discards a growing share of what the first stage does find.** On dev the first stage reaches recall@100 0.7444 and the pipeline delivers 0.6379, so `pool=20` drops 0.1065 -- up from 0.0779 before adoption, because the first stage improved more than the pool can pass through. Raising `pool` is a latency and capacity decision for the answer layer, not a retrieval gain, but it is worth more now than it was.
* **no untouched held-out set.** W3 has been scored twice (see the disclosure above). Cut a fresh window before the next change that needs one.
