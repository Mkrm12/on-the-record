# E1 report: Phase 2 closed, retrieval frozen

Generated 2026-09-27 21:09. Session 15. Sources: `reports/e1_*.json`, `config/retrieval_frozen.json`.

## The decision

**Adopted: nothing.** nothing was established on test and mrr@10, recall@10 point downwards, so nothing is adopted.

Frozen configuration hash: **`e57c6cdfd8793d76`**, recipe `BAAI/bge-small-en-v1.5|full|v2`. Retrieval now changes only for a bug fix, with the dev harness re-run and the change logged.

## Stage A: the two-tier screen on dev

Corpus: 1,422,379 chunks. 108 scorable dev queries.

**Tier 1 cost nothing.** Every arm was scored first on condensed-list metrics, which need no new judgments: unjudged documents are dropped from the ranking rather than assumed irrelevant. An arm that reorders nothing among judged documents cannot have reordered the raw list either, because the condensed list is a subsequence of it.

| arm | dropped at tier 1 | verdict |
|---|---|---|
| B1 | yes | no-op: recall@10 -0.0012, mrr@10 +0.0004 |
| D1 | yes | no-op: recall@10 +0.0000, mrr@10 +0.0002 |
| P30 | no | **PASS** — all four conditions hold |
| P50 | no | **fail** — the sign agrees in only 2 of 4 dev windows (3 required): W0 -0.0011, W1 -0.0352, W2 +0.0065, W4 +0.1365 |
| P50g | no | **fail** — best gain +0.0080 is under the +0.01 floor; the sign agrees in only 1 of 4 dev windows (3 required): W0 +0.0000, W1 -0.0111, W2 +0.0000, W4 +0.0417 |
| R1 | no | **PASS** — all four conditions hold |
| S1 | yes | no-op: recall@10 +0.0019, mrr@10 -0.0008 |

**Tier 2 was judged.** 1,855 unjudged (query, speech) pairs across the baseline and the survivors, **1,820 graded** (35 no reply, 0 unparsable) for **$0.32**. Served model: ['deepseek-flash']. The grades are in `data/uplift/extra_judgments.json` and are unioned in at read time — the committed gold set's own hash is untouched, because every Phase 2 number cites it.

## Stage B: the bundle and drop-one

Candidate bundle after resolving the exclusive groups: **['P30', 'R1']**.
Set aside: none.
Dropped by drop-one (removal moved every metric by < 0.005): **['P30']**.
Final bundle: **['R1']**.

## Stage C: confirmation on the test split, once

| metric | mean | 95% CI | p | adjusted alpha | established |
|---|---|---|---|---|---|
| recall@10 | -0.0084 | [-0.0303, +0.0113] | p = 0.422 | 0.0250 | no |
| mrr@10 | -0.0013 | [-0.0368, +0.0326] | p = 0.9512 | 0.0500 | no |

Paired bootstrap, 10000 resamples, Holm-Bonferroni across the family. A p below the bootstrap floor is reported as an inequality, never as 0.0000.

## The judge

Re-checked on 150 stratified pairs. **Binarised kappa 0.9413** (the gate), four-level 0.8112 (diagnostic; I-77). Verdict: **STABLE**.

Served model: ['deepseek-flash'], fingerprint ['aeb56401ca74e127821c4f9126dcb669']. The judge configuration is frozen from here.

## W5: the sealed exam window

`tests/data/gold_w5.json`. Authored items audited: **65 clean, 2 flagged, 0 blocking** of 67.

Every anchor resolves in W5 and every quote comes out of the database. Adversarial premises were checked against their own evidence, and abstention items were searched across the whole window to test their claim of absence.

## The frozen scorecard

Measured on the frozen configuration over the W5-augmented corpus (1,422,379 embedded chunks), against the pool as extended by Stage A's judging.

| split | queries | R@10 | MRR@10 | nDCG@10 | R@10 condensed | MRR@10 condensed |
|---|---|---|---|---|---|---|
| dev | 108 | 0.4930 | 0.5119 | 0.4321 | 0.4938 | 0.5139 |
| test | 93 | 0.4430 | 0.4250 | 0.3750 | 0.4979 | 0.4963 |

**These are not comparable with the pre-E1 numbers, and the reason matters.** Stage A added 1,820 judgments to the pool. Many found relevant speeches nobody had judged, which enlarges |R| -- the denominator of recall. So dev R@10 reads 0.4930 here against 0.5681 before, while MRR@10 rose from 0.4946 to 0.5119. Recall did not get worse: **the instrument got more honest.** We now know more of what we were missing, and more of what we rank highly is known to be relevant. A number measured against a larger pool is a different number.

The condensed figures are close to the raw ones on dev (0.4938 against 0.4930) and further apart on test (0.4979 against 0.4430), which locates the remaining blind spot: the test split's pool is still the thinner of the two.

## What is not done

- **Incremental judging** for Stage A tier 2. Arms were screened on the coverage the existing pool gives them; an arm below judged@10 0.95 is reported unscorable rather than guessed at.
- **§4.1 items 3, 4, 6, 8, 9, 10** — see `reports/e1_audit.md` §5.
- **Arms C1, D2, Q1, N1, A1, O1, R2** — see `arms.NOT_IMPLEMENTED`.

## Handover to E2

- The **evidence unit** is a speech with its matched sentences (`collapse.Cluster`).
- The **debate ranking** is `arms.rank_debates`, MaxP over the fused pool — used by who_spoke and the overview route.
- The frozen configuration is `config/retrieval_frozen.json`, hash `e57c6cdfd8793d76`. The answer cache must key on it.
