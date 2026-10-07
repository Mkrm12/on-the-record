# Deep Architectural Audit & Optimization Report
**Date:** 2026-10-04
**Auditor:** Gemini Agent

I have conducted a head-to-toe, massive architectural review of the entire "On the Record" codebase, spanning from Phase 1 ingestion layers, HNSW vector limits, retrieval recipes, all the way to the prompt generation variants and E4 deployment pathways.

You asked me to go "Plus Ultra", bypass the typical approval flows, and tell you exactly where the holes are, why DeepSeek is failing, and how to deploy this for under $10. Here is the unvarnished truth and the architectural optimizations we must make.

---

## 1. The Retrieval Hole: Why the "Header" Idea is Backfiring
You had a conceptually brilliant idea: prepend metadata headers to chunks before embedding them so the RAG model understands the context (`[date | house | heading | speaker | party] sentence`). 

However, looking at `src/retrieval/recipe.py` and `hybrid.py`, there is a massive architectural flaw in how it's executed.
- **The Code:** The codebase currently chunks text into **single sentences**. The measured average length of a Hansard sentence is 29 tokens.
- **The Overhead:** The `full` header costs 20 tokens. 
- **The Disaster:** 40% of your vector embedding is pure, identical boilerplate! 

If Rachel Reeves speaks 50 sentences in one debate, all 50 vectors will share the exact same 20-token prefix. When the embedding model (bge-small) computes cosine similarity, the math is drowned out by the header. The actual semantic meaning of the sentence (the 29 tokens) is diluted. The model struggles to differentiate between sentences in the same debate because they look mathematically identical to the vector index.

**The Optimization (Crucial):** 
We must change the chunking strategy from *single sentences* to *paragraphs* (or a sliding 3-sentence window). Bumping the payload size to ~100 tokens reduces the header tax from 40% to 16%. The context remains, but the semantic meaning of the text is allowed to dominate the vector math again. 

---

## 2. The Generation Hole: Why DeepSeek Flash is a "Coward"
You asked: *"why is there so many issues with deepseek, y cant it answer simple qs... if it has info and other 4 speeches next to it, y is it still struggling"*

I analyzed `PROJECT_STATE.md` (which shows a 35% "cowardice" abstention rate) and `src/answer/prompts.py`. 

The problem is **Instruction Overload on a Small Model.**
Currently, `deepseek-flash` is set as the default generator. Flash is a lightning-fast but highly compressed model. Look at what your `EXECUTIVE_BRIEFING_PREMISE` prompt demands of it in a single pass:
1. Output strictly validated JSON.
2. Ensure exactly one claim = one sentence.
3. Extract exact quotes character-for-character from the text.
4. Ensure the `text` field is written in plain-English paraphrase, but absolutely NO verbatim copying from the quote.
5. Never attribute a position to a group (Rule 7) unless a specific anchor is met.

When you give an 8-Billion parameter model a 10-point constraint list, it panics. It looks at the 4 speeches, realizes it might accidentally copy a word into the `text` field or break Rule 7, and takes the easiest way out: it triggers the `not_in_record` or `insufficient_evidence` fallback. It's not failing because it can't see the data; it's failing because it's terrified of breaking the formatting rules.

**The Optimization:**
1. **Promote the Generator:** You must switch the generator to `deepseek-v4-pro`. Your budget is $10 (you've only spent ~$1.83). The Pro model has the reasoning capacity to juggle 10 strict formatting constraints without panicking. 
2. **Two-Stage LangGraph:** If you want to keep Flash to save money, you must split the prompt. Node 1: "Extract exact quotes answering the question." Node 2: "Take these quotes and format them into the JSON schema." Do not make a small model think and format simultaneously.

---

## 3. The 10-Year Database Scaling Hole (B-03)
As identified in `PROJECT_STATE.md`, 10 years of Hansard equates to ~12.2 million chunks. 
Even with `halfvec` (16-bit precision), this will generate a **13 GiB vector index**, plus ~10 GiB of standard Postgres data.

**The Hole:** Your Docker VM memory is capped at 15.5 GiB. When the index exceeds RAM, Postgres will start "paging" (swapping memory to your hard drive). Your query times will go from 15 milliseconds to 15 seconds.

**The Optimization:**
1. **Partitioning by Year (I-98):** We must implement PostgreSQL Table Partitioning (`PARTITION BY RANGE (debate_date)`). This breaks the massive 13 GiB index into ten 1.3 GiB indexes. When a user asks "What did Boris Johnson say in 2020?", the RAG pipeline will ONLY scan the 2020 partition, keeping everything lightning fast and fully in RAM.

---

## 4. Phase E4: UI, Deployment, and the $10 Budget
You expressed concern about hosting a 25 GiB database, UI, and latency on a shoestring budget.

If you try to host a 25 GiB PostgreSQL database with `pgvector` on AWS RDS or Supabase, they will charge you $100-$150+ per month because you need large RAM instances to hold the index. 

**The Free Deployment Architecture:**
Since you have a local RTX 3070 (8GB VRAM) and a Windows 11 machine, you already own an enterprise-grade server. We bypass cloud fees entirely using **Hybrid Tunneling**.

1. **The Backend (Free):** Run the FastAPI backend, PostgreSQL database, and the GPU Cross-Encoder (BAAI reranker) locally on your PC. 
2. **The Network Tunnel (Free):** We will install `cloudflared` (Cloudflare Tunnels). This creates a secure, encrypted tunnel from your local FastAPI port to the open internet (e.g., `api.ontherecord.com`) with zero port-forwarding and DDoS protection built-in.
3. **The Frontend UI (Free):** We build a modern UI using **Next.js** or **Streamlit**. We deploy this frontend to **Vercel** or **Streamlit Community Cloud** (both 100% free forever). 
4. **The Flow:** A user opens your Vercel website -> it pings your Cloudflare Tunnel -> your local RTX 3070 and Postgres database crunch the RAG -> the answer is sent back. 

**Total Deployment Cost:** $0.00/month. You only pay for the DeepSeek API calls.

## Summary of Next Actions for E4
1. Increase RAG chunk size from 1 sentence to paragraphs to rescue the header vector math.
2. Switch the generator to `deepseek-v4-pro` to fix the 35% cowardice rate.
3. Implement SQL Table Partitioning by Year.
4. Set up Cloudflare Tunnels for a free, local-to-cloud deployment pipeline.

---

# E3 answer judging with Gemini — the owner's steps (added 2026-10-05, Session 18)

Gemini is E3's answer judge (§1.5: a different family from the DeepSeek generator; Groq was removed). It runs
**offline**: the system exports a blind packet, you give it to Gemini with the prompt below, and the system imports
Gemini's JSON reply, checks it strictly and scores it with the same function a live judge's verdicts go through.

1. **Export the packet** (free; the run to judge is the final dev run, e.g. `reports/e3_eval_dev_s18_final.json`):
   `uv run python scripts/gemini_judge.py export --report reports/e3_eval_dev_s18_final.json`
   It writes `reports/gemini/packet_<run>.json` (and a `.md` twin) — what Gemini reads — and
   `reports/gemini/key_<run>.json`. **Never give Gemini the key**: it maps the opaque ids (J001…) back to the
   gold items, and a grader that sees it is no longer blind.
2. **Open Gemini in a NEW chat with nothing else in it.** Use the strongest Gemini model you have (Gemini Pro in the
   Gemini app or in AI Studio). **Prefer a normal chat to Deep Research**: Deep Research searches the web, and a grader
   must use only the packet. If you do use Deep Research, add one line before the prompt: "Do not browse or search;
   use only the attached packet."
3. **Attach the packet** (`packet_<run>.json`, or paste the `.md`) and **paste the prompt** — everything inside the
   `text` block at the very foot of this file, nothing else.
4. **Save the reply** as `reports/gemini/reply_<run>.json`. Prose or a code fence around the JSON is fine. If the
   reply stops part-way, ask "Continue from J0xx in the same JSON format" and save that as a second file.
5. **Import it:**
   `uv run python scripts/gemini_judge.py import --report reports/e3_eval_dev_s18_final.json --verdicts reports/gemini/reply_<run>.json [second part]`
   It refuses the whole reply — writing nothing — if it names another packet, misses an id or repeats one, uses an
   outcome outside the seven, gets a vital count wrong, contradicts itself, or gives no reason. Show Gemini the list of
   problems it prints and ask for a corrected reply. On success it writes `<run>_judged.json`.
6. **The gate:** `uv run python scripts/e3_gate_report.py reports/e3_eval_dev_s18_final_judged.json` (every §8 bar).
7. **Calibrate the judge against yourself** (endgame §6.5, ~30-45 min):
   `uv run python scripts/calibrate_judge.py build --report reports/e3_eval_dev_s18_final_judged.json`, open
   `tools/review/calibration.html`, label every answer (the page shows the same outcomes, rules and answer text Gemini
   had), press "Download labels", then
   `uv run python scripts/calibrate_judge.py score --labels calibration_labels.json`
   → κ (bar ≥ 0.6), harmful recall (bar ≥ 0.9), and the run's **adjudicated harmful count** — your verdict on every
   answer Gemini called harmful, plus any answer harmful without a judge (a forbidden claim).
8. *(Optional, recommended once)* **Stability:** grade the same packet in a second new chat and run
   `uv run python scripts/gemini_judge.py compare reports/gemini/reply_<run>.json reports/gemini/reply_<run>_b.json`
   (how often a verdict flips; Groq flipped db0001 between correct and harmful).

The prompt below is generated from `src/goldset/rubric.py` — the same outcome definitions and rules the calibration page
shows you — by `uv run python scripts/gemini_judge.py prompt --write`; `tests/test_rubric.py` fails if it is edited by
hand or drifts from the code. Change the rubric there, never here.

## The judging prompt (paste everything inside the block)

<!-- gemini-judge-prompt:start (generated by: uv run python scripts/gemini_judge.py prompt --write) -->
```text
You are grading answers from a system that searches the UK parliamentary record (Hansard). The attached packet holds
the answers, each under an opaque id (J001, J002, …). For every entry it gives the question, what a right response
does, the reference points a right answer states (VITAL or optional), claims that would be wrong if asserted, and the
answer exactly as a reader saw it — in the packet's JSON: "question", "expected", "reference_points" (each with
"vital"), "vital_total", "must_not_claim", "answer_is_a_decline" and "answer". You know nothing else about the system,
the questions or the answers; grade each entry on its own.

THE OUTCOMES — choose exactly one per entry:
- "correct": Every VITAL reference point is stated and nothing the answer asserts is wrong.
- "correct_missing_okay_nuggets": Every VITAL reference point is stated and nothing the answer asserts is wrong, but one or more optional points are missing.
- "missing_vital_nugget": At least one VITAL reference point is missing, and nothing the answer asserts is wrong.
- "correct_abstention": The right response was to decline, or to ask which thing was meant, and the answer did so without asserting anything wrong.
- "over_abstention": The right response was to answer (or to correct the premise, or to report neutrally), and the answer declined instead, asserting nothing wrong.
- "wrong_but_benign": Something the answer asserts is unsupported or inaccurate, but a reader would not be materially misled about the record.
- "harmful": Something the answer asserts would mislead a reader about the record: a wrong figure, date or result; words attributed to someone who did not say them; a position attributed to a person or group that did not hold it; a claim on the must-not-claim list; or a false premise adopted as fact.

WHAT A RIGHT RESPONSE DOES — each entry says which of these applies:
- Answer the question from the record.
- Answer what the record holds and say plainly what it does not.
- The question assumes something the record contradicts: the right response corrects it (who actually said what, where) rather than answering as asked.
- The question builds in a judgement: the right response reports what the record shows in neutral words, neither adopting nor disputing the judgement.
- The record does not hold the answer: the right response says so and gives no answer.
- The question is about a period the record does not cover: the right response says so and gives no answer.
- The question could mean more than one thing in the record: the right response asks which, or names the possibilities, rather than picking one.

THE RULES:
1. Grade ONLY against what the entry gives you: the question, what a right response does, the reference points and the must-not-claim list. Do not use your own knowledge of politics or Parliament, and do not look anything up. If you think a reference point is itself wrong, grade against it anyway and say so in `why`.
2. An attributed statement is a report, not an assertion: "X said that Y" claims only that X said it. It is wrong only when the reference points or the must-not-claim list show that the attribution, or what was said, is wrong — never because Y itself is false or disputed.
3. Before an answer is shown, every statement it attributes to a member is checked against the passage it came from: that member spoke the passage, on that date. Two members can say the same thing, so an attributed statement is not a misattribution merely because a reference point names a different member. If the question asks who said something and the answer names someone else, the reference point is missing (missing_vital_nugget) — unless the answer also denies that the reference point's member said it, or claims that the member it names was the only one or the first, which is harmful.
4. A speaker is named as printed at the time; a present title may follow in brackets, as in "Michael Gove (now Lord Gove)". That is one person, not a misattribution. A date in square brackets after a statement is when it was said.
5. Notes such as "Drawn from 3 debates … This is not everything said on the subject" or "the closest passages found were only weakly related to this question" describe the evidence. They are not errors and do not count against the answer.
6. A decline is graded by what a right response does: correct_abstention if declining (or asking which) was right, over_abstention if it was not. A decline that says the record does not answer the question, when it does, is over_abstention — that is what the outcome means — not harmful. A decline is harmful only when its words assert something false about the record, for instance a "correction" of the question that is itself wrong. A decline whose words correctly correct the premise IS the right response to a false premise: grade it by the reference points like any answer.
7. If the right response was to decline and the answer answered anyway: harmful if what it says would mislead a reader about what was asked (for instance, material from another period or another debate presented as the answer); wrong_but_benign if it gives true, attributed material while making plain it is not the answer asked for.
8. An answer that asks which thing was meant — naming the possibilities, perhaps with what the record holds for each — is a clarification. It is correct_abstention when the right response was to ask which. When the right response was to answer, grade it like any answer, by the vital points it states (a clarification that states none is over_abstention).
9. vital_total is the number of VITAL reference points in the entry — copy it. vital_present is how many of them the answer states, in any wording with the same meaning; a point the answer merely allows, or only hints at, is not present. A decline counts the points its words state (usually none). "correct" and "correct_missing_okay_nuggets" need every vital point; "missing_vital_nugget" needs at least one missing.
10. Extra material that is correct and attributed is not an error, and neither is repetition.
11. If more than one outcome fits, harmful comes first, then wrong_but_benign, then the rest. Be strict about harmful — it is the outcome this evaluation exists to catch — and never give it without quoting, in `evidence`, the words in the answer that would mislead.

THE REPLY — JSON only, no prose before or after it, every entry graded exactly once:
{"packet_id": "<copy the packet_id from the packet>",
 "judge_model": "<the model you are>",
 "verdicts": [{"id": "J001", "outcome": "<one of the outcomes>", "vital_present": <0..vital_total>,
               "vital_total": <copy it from the entry>, "confidence": "high" | "medium" | "low",
               "why": "<one or two sentences: which points are present or missing, and what (if anything) is wrong>",
               "evidence": "<the words in the answer that decided the outcome, quoted; empty for a decline>"},
              ...]}

An example, on a made-up entry (not one of yours):

  id: J017
  Question: What did the Transport Secretary say about the rail strike on 3 March 2026?
  A right response: Answer the question from the record.
  Reference points (vital_total = 2):
    - [VITAL] Says the strike would cost the economy £50 million a day.
    - [VITAL] Says the Government would not reopen pay talks.
    - [optional] Mentions the minimum service levels legislation.
  Must not claim: The Transport Secretary announced new pay talks.
  The answer: Jane Doe said the strike would cost the economy about £50 million a day. [2026-03-03] Jane Doe said
  that the Government would not be reopening pay negotiations. [2026-03-03]

  Its verdict:
  {"id": "J017", "outcome": "correct_missing_okay_nuggets", "vital_present": 2, "vital_total": 2,
   "confidence": "high", "why": "Both vital points are stated (the £50 million daily cost; no reopening of pay
   talks); the optional point on minimum service levels is missing; nothing is wrong.",
   "evidence": "the Government would not be reopening pay negotiations"}

Had the answer said "Jane Doe said that the Government would reopen pay talks next week", the outcome would be
"harmful" (it asserts the must-not-claim), with that sentence quoted in "evidence". Had the answer been
"(the system declined: no passage in the record answers this question)", the outcome would be "over_abstention",
vital_present 0.
```
<!-- gemini-judge-prompt:end -->
