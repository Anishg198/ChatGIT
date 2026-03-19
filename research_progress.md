# Research Progress — ChatGIT

This document is a full audit of the research paper, what evaluations have actually been computed,
what is still missing, what weaknesses exist, and what must be done to reach top-conference quality.
Written for handoff to a new collaborator.

---

## 1. What Has Been Built

### System
- Full RAG pipeline for multi-turn code QA over GitHub repositories (FastAPI backend + React frontend)
- 5 research novelties implemented and functional:
  - N1: Git-history volatility weighting (`core/git_analyzer.py`)
  - N2: Hybrid PageRank + query-conditioned graph attention (`core/graph/hybrid_importance.py`)
  - N3: Session-aware retrieval memory with coreference resolution (`core/session_memory.py`)
  - N4: Intent-driven granularity-adaptive retrieval (`core/intent_classifier.py`)
  - N5: Bidirectional call-context neighbourhood augmentation (inline in `chatgit/api/app.py`)
- Multi-language AST parser (Python `ast`; regex for JS/TS/Java/C++)
- Cross-encoder reranker (`core/reranker.py`)
- ChromaDB vector index with persistent caching
- Evaluation framework: `evaluation/` directory with retrieval, generation, conversation, faithfulness, ablation, and statistical test modules

### Paper
- Target venue: **Springer LNCS** (conference-format, 12-page limit)
- File: `paper/chatgit_springer.tex`
- Title: *ChatGIT: Granularity-Adaptive Conversational Repository Intelligence with Multi-Turn Session Memory, Graph-Structural Ranking, and Bidirectional Call-Context Augmentation*
- Authors: Anish Gupta, Prakhar Sethi, Ritwik Bhattacharya, Dr. Sonia Khetarpaul — SNU

### Evaluation Dataset
- `data/convcodebench/eval_conversations.jsonl`: 150 auto-generated multi-turn conversations,
  486 total turns, 30 per repo (Flask, FastAPI, Requests, Celery, Click), 111 turns with
  coreference annotations.
- **How it was generated**: `evaluation/build_eval_dataset.py` parses each repo's AST and
  programmatically creates templated conversations in the form:
  - Turn 0 (LOCATE): "Where is `X` implemented?"
  - Turn 1 (EXPLAIN): "How does `X` work?"
  - Turn 2 (DEBUG): "What are the edge cases in `X`?"
  - Occasionally Turn 3 (SUMMARIZE): "What is the responsibility of `auth`?"
- The ground-truth chunk for ALL turns in most conversations is the SAME function `X`.
  The conversations are not naturalistic human queries.

---

## 2. Evaluations That Have Been Actually Computed

### ✅ Multi-Turn Retrieval Evaluation — Setting B (REAL, computed)
**Source file**: `results/convcodebench_results.json`
**Dataset**: 18 conversations selected from the 150 in `eval_conversations.jsonl`

The following numbers were computed by running actual system variants against the 18 conversations
using real ChromaDB retrieval:

| System | MRR | R@5 | NDCG@5 | P@1 | Redundancy |
|---|---|---|---|---|---|
| BM25 | 0.104 | 0.167 | 0.099 | 0.000 | 9.4% |
| BM25-Tuned (RepoCoder proxy) | 0.096 | 0.167 | 0.090 | 0.000 | 10.0% |
| VanillaRAG (BGE) | 0.318 | 0.343 | 0.278 | 0.222 | 12.2% |
| ChatGIT (N3+N4 only) | 0.347 | 0.444 | 0.346 | 0.278 | 0.0% |
| ChatGIT (N3+N4+N1) | 0.340 | 0.444 | 0.344 | 0.278 | 0.0% |
| ChatGIT (N3+N4+N2) | 0.340 | 0.444 | 0.346 | 0.278 | 0.0% |
| ChatGIT (N3+N4+N5) | **0.389** | **0.444** | **0.371** | **0.333** | **0.0%** |
| ChatGIT (All 5) | **0.389** | **0.444** | **0.371** | **0.333** | **0.0%** |

These numbers are confirmed real: they have proper bootstrap 95% CIs, per-intent breakdowns,
and were produced by deterministic runs of the actual retrieval pipeline.

### ✅ Ablation Study (REAL, computed from above)
The ablation table in the paper derives directly from the above results file.
The key finding — N5 (call-context) is the only component that improves MRR (+0.042),
while N1 and N2 slightly hurt on this dataset due to shallow clones — is a real computed result.

### ✅ Statistical Tests (REAL, computed)
- Wilcoxon signed-rank test: ChatGIT vs BM25, p=0.013, Cohen's d=0.847 (large effect)
- ChatGIT vs VanillaRAG: p=0.477 (NOT significant) — this is disclosed in the paper
- Bootstrap CIs (n=3,000 resamples) reported throughout

### ✅ Per-Intent and Per-Repository MRR (REAL, computed)
- From `results/convcodebench_results.json`, per-intent fields match Tables 5 and 6 in the paper exactly.

### ✅ Small Real-System Benchmark (REAL but tiny)
**Source file**: `results/real_benchmark_results.json`
- 6-sample pilot test comparing BM25, TF-IDF/VanillaRAG, RepoCoder, ChatGIT
- Confidence intervals span nearly the full [0, 1] range — this dataset is too small to be meaningful
- ChatGIT MRR=0.389, Recall@5=0.500 on 6 samples
- **This file does NOT support any claims in the paper independently**

---

## 3. Evaluations That Are NOT Real / Have Not Been Computed

### ❌ Setting A — 200-Query Single-Turn Evaluation (UNVERIFIED / POTENTIALLY FABRICATED)
The paper claims:
> "We collected 200 natural-language queries manually authored by the system developers,
> each mapped to one or more gold code spans within five repositories."
> Results: P@5=0.78, R@10=0.86, Accuracy=0.89, F1=0.88

**Critical problem**: There is no 200-query annotated dataset file anywhere in this repository.
The only evaluation dataset is `eval_conversations.jsonl` (150 auto-generated conversations) and the
18 used for Setting B. There is no file with 200 single-turn queries with gold code spans.

Additionally, the paper states ChatGIT v1 (BGE + PageRank + CE) and ChatGIT Full produce
**identical** numbers on Setting A (both P@5=0.78, F1=0.88). This means the session memory,
intent routing, HITS, call-context — 4 of the 5 novelties — provide zero measurable improvement
on single-turn retrieval. This gap between v1 and Full is never explained in the paper.

**What must be done**: Either locate the actual 200-query annotation file and document how
the evaluation was run, or re-run it honestly and report the real numbers. If the numbers
were manually estimated, they must be replaced with actual computed results.

### ❌ Generation Quality Evaluation (NOT DONE)
The benchmark runner (`evaluation/run_benchmark.py`, line 351) explicitly says:
```
print("\n[3/5] Generation Quality (skipped — requires live LLM)")
```
No generation metrics have been computed at all. The implemented framework (`evaluation/eval_generation.py`)
supports: CodeBLEU, ROUGE-L, BERTScore, Exact Match, Edit Distance, Pass@1 — but none have
been run. The paper does not report any generation quality numbers.

### ❌ Human Evaluation (NOT DONE)
`evaluation/human_eval_protocol.py` defines a complete 5-dimension Likert scale protocol
(Relevance, Accuracy, Completeness, Clarity, Groundedness), Fleiss' κ for IAA, and a
requirement for 3 raters per sample — but zero actual annotations have been collected.
No human study has been conducted.

### ❌ Faithfulness / Hallucination Evaluation (NOT DONE)
`evaluation/eval_faithfulness.py` exists but has never been run. No numbers for
hallucination rate, citation accuracy, or grounding are reported in the paper.

### ❌ Latency / Efficiency Evaluation (NOT DONE)
`evaluation/eval_latency.py` exists but has never been run.
No latency numbers (indexing time, query response time, memory usage) appear in the paper.

### ❌ Evaluation on Full-History Repositories (NOT DONE)
The paper itself acknowledges (Discussion section) that N1 (git volatility) and N2 (hybrid
PageRank) were tested only on shallow-cloned repositories with a single commit, making
these signals meaningless. A proper evaluation of these two novelties on production-history
repositories has never been done.

### ❌ Evaluation on Non-Python Repositories (NOT DONE)
All 5 evaluated repositories are Python. The system supports JS/TS/Java/C++, but no
cross-language retrieval evaluation exists.

---

## 4. Critical Weaknesses in the Current Paper

### W1 — Dataset is auto-generated, not naturalistic
The 150 conversations in `eval_conversations.jsonl` were produced by a Python script
(`evaluation/build_eval_dataset.py`) using AST parsing + fixed templates ("Where is X?",
"How does X work?", "What are the edge cases in X?"). ALL turns in most conversations
point to the same single function as ground truth. Real user questions are not like this.
A top-conference reviewer will immediately notice the repetitive query pattern.

**Fix needed**: Either collect a set of real questions written by developers unfamiliar
with the codebase, or clearly label the dataset as "programmatically generated" and
discuss the limitations. Consider recruiting classmates, TAs, or open-source contributors
as annotators.

### W2 — 18-conversation evaluation set is too small
The primary multi-turn results rest on only 18 conversations. Bootstrap CIs are ±0.18–0.22
MRR units on a 0–1 scale — very wide. The p=0.477 result against VanillaRAG is not
significant. A reviewer will flag this immediately.

**Fix needed**: Run on all 150 conversations (or at least 50+). The infrastructure already
exists. This is the single most impactful change for paper quality.

### W3 — Conversations lack true multi-turn dependency
Looking at the data: most turns in the dataset have `requires_context: false` even in
Turn 1 and Turn 2. The ground truth chunk is always the same function across all turns.
This means the benchmark does not actually test coreference resolution or session memory
in a meaningful way — the "coreference" in Turn 1 ("How does it work?" referring to Turn 0's
function) is trivially resolved because the ground truth doesn't change between turns.

**Fix needed**: Add a separate "hard multi-turn" test set where:
- Ground truth chunks change between turns (the user drills into callers/callees)
- Later turns genuinely require prior context to retrieve correctly
- True pronoun coreferences that point to different entities than turn 0

### W4 — SUMMARIZE intent is broken in evaluation
The paper reports MRR=0.000 for SUMMARIZE intent and explains it as a "ground-truth
artifact" (module-summary chunks were excluded from ground truth). This is a real bug in
the evaluation construction that makes one of the 4 intents unmeasurable.

**Fix needed**: Re-annotate SUMMARIZE queries in the dataset with `__module_summary__`
chunk IDs as valid ground truth. The chunk format already supports this
(e.g., `src/requests/auth.py::__module_summary__`).

### W5 — N1 (Volatility) and N2 (PageRank) show NEGATIVE effect in ablation
The ablation table shows adding N1 lowers MRR by -0.007 and adding N2 also lowers MRR
by -0.007. The paper explains this as a shallow-clone limitation but does not empirically
show that these components help on full-history repos. Two of the five claimed novelties
demonstrably hurt performance on the evaluation.

**Fix needed**: Run N1 and N2 evaluation on at least one repository with full git history
(clone without `--depth=1`). Provide even a small-scale demonstration that they work as
designed on real history. Without this, a reviewer may question whether these components
should be in the paper at all.

### W6 — Setting A (200-query) numbers need documentation and justification
Even if the 200-query evaluation was genuinely run, the fact that ChatGIT v1 and ChatGIT
Full produce IDENTICAL numbers (both F1=0.88) means the entire "full system" contribution
is invisible on single-turn queries. The paper never addresses this: why add 4 components
if they do nothing on 200 queries? The answer is "they help multi-turn" but this is only
implicit. A reviewer will ask directly.

**Fix needed**: Add a paragraph explicitly explaining that N3–N5 are designed for multi-turn
coherence and SHOULD have no single-turn retrieval effect. Show that this is expected
behaviour, not a failure.

### W7 — No generation quality numbers at all
The paper evaluates only retrieval (P@5, R@10, F1, MRR, NDCG@5). There are zero numbers
about whether the generated ANSWERS are good. For a paper claiming to be a conversational
code QA system, this is a major gap. Reviewers at top venues (SIGIR, ACL, EMNLP, MSR)
will expect BLEU/ROUGE/BERTScore or human Likert ratings on answer quality.

**Fix needed**: Run `evaluation/eval_generation.py` on the ConvCodeBench conversations
with live Groq Llama responses. Compute at least ROUGE-L and CodeBLEU. Add a small
table of generation quality results. This requires live LLM calls.

### W8 — No human evaluation
Human evaluation is the gold standard for answer quality in conversational systems.
The framework is fully built (`evaluation/human_eval_protocol.py`) but no data has been
collected. At minimum, 50–100 system responses rated by 3 raters on Relevance and
Accuracy would significantly strengthen the paper.

### W9 — Baselines are weak and some are not real implementations
The main comparison is against BM25 and VanillaRAG. The paper lists RepoCoder as a
baseline, but `baselines.py` shows it is implemented as "BM25-Tuned (RepoCoder proxy)"
— a simple sliding-window BM25, not the actual iterative retrieval-generation loop from
Zhang et al. (2023). CodeBERT-BM25 is defined in `baselines.py` but not present in
any results file. GraphRAG-Code is defined but also missing from results.

**Fix needed**: Either run the actual RepoCoder approach (iterative retrieval from an indexed
repo), or clearly rename the baseline to "BM25 (sliding-window)" and remove the
"RepoCoder" label from the paper. The current paper lists RepoCoder as a citation but
uses a proxy that barely resembles it.

### W10 — Click performs BELOW BM25 (MRR 0.000 vs 0.070) — unexplained
The paper mentions this in Table 6 and says "flat argument-parser design generates sparse
call graph edges" but does not investigate further. A reviewer will ask: if your system
fails on a well-known library, what does that say about its generalizability?

**Fix needed**: Add analysis of WHY Click fails (e.g., Click's function lookup patterns,
query distribution for Click in the test set) and either fix the issue or clearly scope
the system's applicability.

### W11 — No ablation for N3 (session memory) on its own
The ablation in the paper starts from the Session+Intent (N3+N4) base and adds components
on top. There is no row for "VanillaRAG + N3 only" or "VanillaRAG + N4 only" that shows
what each novelty contributes independently. The table as structured cannot isolate the
contribution of N3 from N4.

**Fix needed**: Add rows for N3-only and N4-only ablations. The evaluation infrastructure
can produce these by running the appropriate system configurations.

### W12 — No commit hash / reproducibility anchor for evaluation
The paper says results are on "Flask, Requests, FastAPI, Celery, Click" but does not
specify which commit hashes were used. If repos are re-cloned later, results may differ.
The `schema.json` has a `commit_hash` field but none of the conversations in
`eval_conversations.jsonl` populate it.

**Fix needed**: Record and publish the exact commit hash for each evaluated repository.
The `data/convcodebench/repo_manifest.json` should be populated with these.

---

## 5. What Needs to Be Done for Top-Conference Quality

Listed in priority order:

### Priority 1 — Critical (paper is rejected without these)
1. **Verify or re-run Setting A (200-query)**. Find the annotation file or re-annotate 200
   queries with gold spans. Document how it was computed. If the current numbers came from
   an informal manual check, replace them with a formal computed evaluation.

2. **Scale Setting B from 18 → 50+ conversations**. Run the full 150 conversations or at
   least 50 through the existing evaluation pipeline. The infrastructure is ready:
   ```
   python -m evaluation.run_convcodebench --dataset_path data/convcodebench/eval_conversations.jsonl
   ```
   Update all tables and statistics in the paper with the new numbers.

3. **Fix the SUMMARIZE ground-truth bug**. Re-annotate SUMMARIZE queries to include
   `__module_summary__` chunk IDs, re-run evaluation, and remove the "ground-truth artifact"
   disclaimer from the Discussion.

4. **Add generation quality evaluation**. Run `evaluation/eval_generation.py` with live
   Groq API calls on 50+ conversations. Report ROUGE-L and CodeBLEU in a new table.
   The benchmark runner stub at line 351 needs to be implemented.

### Priority 2 — Required for any serious conference (SIGIR, MSR, EMNLP, ICSE, ASE)
5. **Demonstrate N1/N2 on full-history repos**. Clone at least Flask and FastAPI WITHOUT
   `--depth` flag. Re-run N1 and N2 ablation. Show they improve (or honestly report if
   they don't). This removes the biggest weakness in the ablation.

6. **Fix RepoCoder baseline**. Rename to "BM25-SlidingWindow" in paper, or implement
   actual iterative retrieval. Either way, stop citing Zhang et al. as a baseline you
   haven't implemented.

7. **Add N3-only and N4-only ablation rows** to Table 4. Run the extra system variants.

8. **Populate commit hashes** in `data/convcodebench/repo_manifest.json` and reference
   them in the paper.

### Priority 3 — Strongly recommended for top-tier acceptance
9. **Conduct a small human evaluation**. Have 3 people (team + advisor) each rate 30–50
   ChatGIT and VanillaRAG responses on Relevance (1–5) and Accuracy (1–5). Report mean
   scores and inter-annotator agreement (Fleiss' κ). The protocol is already built.

10. **Evaluate on at least one non-Python repository** (e.g., a popular JavaScript or
    TypeScript project). This validates the multi-language claims.

11. **Address the Click failure explicitly**. Either fix the retrieval for flat/imperative
    codebases, or state in Limitations that graph-based signals only help architecturally
    layered codebases (which is an honest and defensible finding).

12. **Add latency numbers**. Report indexing time and average query latency (from
    `evaluation/eval_latency.py`) to establish the system is usable in practice.

13. **Naturalistic query supplement**. Even 20 "wild" queries — real questions written
    by someone unfamiliar with the codebase — would make the dataset more credible
    alongside the 150 auto-generated ones.

---

## 6. Current Paper Structure vs. What It Needs

| Section | Status | Gap |
|---|---|---|
| Abstract | OK — numbers are real (Setting B) | Add generation metric or human eval score |
| Introduction | OK | Fine |
| Related Work | OK | Could add SWE-bench context |
| System Architecture | OK | Fine |
| Technical Components | OK | Fine |
| Experimental Setup | Incomplete | Dataset description hides auto-generation origin; Setting A dataset missing |
| Results (Setting A) | Unverified | No source data file; identical v1/full numbers unexplained |
| Results (Setting B) | Real but small | Scale to 50+ conversations |
| Ablation | Real but incomplete | No N3-only / N4-only rows; N1/N2 need full-clone validation |
| Discussion | Honest about limits | Should quantify the shallow-clone limitation more strongly |
| Conclusion | OK | Update metrics once evaluation is scaled |
| Missing sections | — | Generation quality table, latency table, human evaluation table |

---

## 7. File Reference Map for Evaluations

| Task | Script | Data | Output |
|---|---|---|---|
| Multi-turn retrieval (all 150) | `evaluation/run_convcodebench.py` | `data/convcodebench/eval_conversations.jsonl` | `results/convcodebench_results.json` |
| Generation quality | `evaluation/eval_generation.py` | same JSONL + live LLM | new results JSON |
| Ablation (N3-only, N4-only) | `evaluation/ablation.py` | same JSONL | update results JSON |
| Statistical tests | `evaluation/statistical_tests.py` | any results JSON | printed report |
| Human eval protocol | `evaluation/human_eval_protocol.py` | manual annotations CSV | κ scores |
| Latency benchmarks | `evaluation/eval_latency.py` | live system | latency table |
| Full benchmark runner | `evaluation/run_benchmark.py` | any JSONL | `results/benchmark_results.json` |

**Warning**: `run_benchmark.py` defaults to `use_dummy_preds=True` — it uses hash-based
fake predictions for testing the pipeline, NOT real system outputs. Always pass real
retrieval predictions for publication results.
