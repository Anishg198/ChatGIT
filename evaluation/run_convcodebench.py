"""
Full ConvCodeBench evaluation — 6 Python repos, all 9 conversations.
Uses simple token counter to avoid tiktoken subprocess hangs.
"""

import sys, json, os, time
sys.path.insert(0, '/Users/anishgupta/Desktop/ChatGIT')

# ── Patch tiktoken BEFORE importing chunker ───────────────────────────────────
import chatgit.core.chunker as _ck
_ck._count_tokens = lambda text: len(text) // 4   # simple fallback, no subprocesses
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from chatgit.core.chunker import chunk_repository
from chatgit.core.intent_classifier import classify_intent
from chatgit.core.session_memory import SessionRetrievalMemory
from chatgit.core.git_analyzer import GitVolatilityAnalyzer          # N1
from chatgit.core.graph.pagerank import CodePageRankAnalyzer          # N2 (graph)
from chatgit.core.graph.hybrid_importance import HybridImportanceScorer  # N2
from evaluation.baselines import BM25, TFIDFRetriever, RepoCoderStyle
from evaluation.eval_retrieval import evaluate_retrieval, print_retrieval_report
from evaluation.statistical_tests import full_comparison_report, print_comparison_table

REPOS = {
    "flask":    "/tmp/flask_bench",
    "requests": "/tmp/requests_bench",
    "click":    "/tmp/click_bench",
    "fastapi":  "/tmp/fastapi_bench",
    "celery":   "/tmp/celery_bench",
}

CONVERSATIONS_PATH = "data/convcodebench/sample_conversations.jsonl"


# ── Chunk all repos ───────────────────────────────────────────────────────────

SKIP_DIRS_EXTRA = {"tests", "test", "docs", "doc", "examples", "example",
                   "benchmarks", "scripts", "contrib", "extras", "tools"}

def chunk_all_repos():
    all_chunks = {}
    for repo_id, path in REPOS.items():
        print(f"  Chunking {repo_id}...", end=" ", flush=True)
        t0 = time.time()
        docs = chunk_repository(path)
        chunks, seen_ids = [], {}
        for d in docs:
            fname = d.metadata['file_name']
            # Skip test/doc dirs to keep chunk count manageable
            parts = set(fname.replace("\\","/").split("/"))
            if parts & SKIP_DIRS_EXTRA:
                continue
            cid  = f"{fname}::{d.metadata['node_name']}"
            text = d.text if hasattr(d, 'text') else d.page_content
            obj  = {"id": cid, "text": text,
                    "file":      fname,
                    "node_type": d.metadata.get('node_type', ''),
                    "node_name": d.metadata.get('node_name', '')}
            chunks.append(obj)
            if cid not in seen_ids:
                seen_ids[cid] = obj
        all_chunks[repo_id] = {"list": chunks, "by_id": seen_ids}
        print(f"{len(chunks)} src chunks  ({time.time()-t0:.1f}s)")
    return all_chunks


# ── GT fuzzy matching ─────────────────────────────────────────────────────────

def fuzzy_match_gt(gt_id: str, chunk_by_id: dict) -> list:
    if gt_id in chunk_by_id:
        return [gt_id]
    parts = gt_id.split("::")
    if len(parts) != 2:
        return []
    file_part, name_part = parts
    bare_name = name_part.split(".")[-1]   # strip "ClassName." prefix
    candidates = []
    for cid in chunk_by_id:
        if "::" not in cid:
            continue
        cfile, cname = cid.split("::", 1)
        file_match = (file_part in cfile) or (cfile in file_part) or \
                     (cfile.split("/")[-1] == file_part.split("/")[-1])
        if not file_match:
            continue
        if cname == bare_name or cname == name_part:
            candidates.append(cid)
    if not candidates:
        for cid in chunk_by_id:
            cname = cid.split("::")[-1] if "::" in cid else ""
            if cname == bare_name and len(bare_name) > 3:
                candidates.append(cid)
    return candidates


# ── Load conversations ────────────────────────────────────────────────────────

def load_conversations(all_chunks):
    convs = []
    with open(CONVERSATIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(json.loads(line))

    queries, sessions = [], []
    gt_hit = gt_miss = 0

    for conv in convs:
        repo_id = conv["repo_id"]
        if repo_id not in all_chunks:
            continue
        chunk_by_id = all_chunks[repo_id]["by_id"]
        session_turns = []
        for turn in conv["turns"]:
            matched_gt = []
            for g in turn.get("ground_truth_chunks", []):
                m = fuzzy_match_gt(g, chunk_by_id)
                if m:
                    matched_gt.extend(m);  gt_hit += 1
                else:
                    gt_miss += 1
            # dedup
            seen = set()
            matched_gt = [x for x in matched_gt if not (x in seen or seen.add(x))]

            qid = f"{conv['conversation_id']}_t{turn['turn_id']}"
            row = (qid, turn["query"], matched_gt,
                   turn.get("intent","unknown"), repo_id,
                   conv["conversation_id"])
            queries.append(row)
            session_turns.append({"qid": qid, "query": turn["query"],
                                   "gt": matched_gt,
                                   "intent": turn.get("intent","unknown")})
        sessions.append({"conv_id": conv["conversation_id"],
                          "repo_id": repo_id, "turns": session_turns})

    total = gt_hit + gt_miss
    print(f"  GT mapping: {gt_hit}/{total} matched ({100*gt_hit/max(total,1):.0f}%)")
    return queries, sessions


# ── Retrievers ────────────────────────────────────────────────────────────────

def build_retrievers(all_chunks, embed_model):
    retrievers = {}
    for repo_id, data in all_chunks.items():
        chunks = data["list"]
        repo_path = REPOS[repo_id]
        print(f"  {repo_id}: {len(chunks)} chunks — BM25 + BGE...",
              end=" ", flush=True)
        t0 = time.time()
        # Truncate text for all retrievers to keep fitting fast
        trunc = [{"id": c["id"], "text": c["text"][:300],
                  "file": c["file"], "node_type": c["node_type"],
                  "node_name": c["node_name"]} for c in chunks]
        bm25  = BM25().fit(trunc)
        bm25b = BM25(k1=1.2, b=0.5).fit(trunc)
        texts = [c["text"][:300] for c in chunks]
        embs  = embed_model.encode(texts, batch_size=256, show_progress_bar=False,
                                    normalize_embeddings=True).astype(np.float32)

        # ── N1: git volatility analysis ──────────────────────────────────
        git_analyzer = GitVolatilityAnalyzer()
        git_analyzer.analyze(repo_path)   # silent if no git history

        # ── N2: PageRank + hybrid importance scorer ──────────────────────
        pagerank = CodePageRankAnalyzer()
        try:
            pagerank.analyze_repository(repo_path)
            pr_dict = dict(pagerank.get_function_pagerank())
            hybrid_scorer = HybridImportanceScorer(pagerank.function_graph)
            # Build node embeddings using SentenceTransformer directly
            nodes_to_embed = sorted(pr_dict, key=pr_dict.get, reverse=True)[:500]
            short_names = [n.split("::")[-1] for n in nodes_to_embed]
            node_embs_raw = embed_model.encode(short_names, batch_size=256,
                                               show_progress_bar=False,
                                               normalize_embeddings=True)
            hybrid_scorer._pagerank = pr_dict
            hybrid_scorer._node_embs = {
                node: np.array(emb)
                for node, emb in zip(nodes_to_embed, node_embs_raw)
            }
            hybrid_scorer._built = True
        except Exception as e:
            print(f"\n  [N2 warn] {e}", end=" ")
            pagerank = None
            hybrid_scorer = HybridImportanceScorer(None)

        retrievers[repo_id] = {
            "bm25": bm25, "bm25_tuned": bm25b,
            "chunks": chunks, "embs": embs,
            "git_analyzer": git_analyzer,          # N1
            "hybrid_scorer": hybrid_scorer,        # N2
            "pagerank": pagerank,                  # N2 (call graph for N5)
        }
        print(f"{time.time()-t0:.1f}s")
    return retrievers


def run_dense(retrievers, queries, embed_model, k=10):
    """VanillaRAG: pure BGE cosine similarity, no session memory, no intent routing."""
    preds = []
    for qid, query, gt, intent, repo_id, _ in queries:
        if repo_id not in retrievers or not gt:
            continue
        rv    = retrievers[repo_id]
        q_emb = embed_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        sims  = rv["embs"] @ q_emb
        top   = np.argsort(-sims)[:k]
        retrieved = [rv["chunks"][i]["id"] for i in top]
        preds.append({"query_id": qid, "retrieved": retrieved,
                      "ground_truth": gt, "intent": intent})
    return preds


def run_lexical(key, retrievers, queries, k=10):
    preds = []
    for qid, query, gt, intent, repo_id, _ in queries:
        if repo_id not in retrievers or not gt:
            continue
        retrieved = retrievers[repo_id][key].retrieve_ids(query, k=k)
        preds.append({"query_id": qid, "retrieved": retrieved,
                      "ground_truth": gt, "intent": intent})
    return preds


def run_chatgit(retrievers, queries, embed_model, k=10):
    by_conv = defaultdict(list)
    for row in queries:
        by_conv[row[5]].append(row)

    preds = []
    for conv_id, conv_rows in by_conv.items():
        repo_id = conv_rows[0][4]
        if repo_id not in retrievers:
            continue
        rv     = retrievers[repo_id]
        chunks = rv["chunks"]
        embs   = rv["embs"]
        session_mem = SessionRetrievalMemory()

        for qid, query, gt, intent, _, _ in conv_rows:
            if not gt:
                continue
            cfg = classify_intent(query)
            resolved = session_mem.resolve_coreferences(query)
            q_emb = embed_model.encode([resolved], normalize_embeddings=True)[0].astype(np.float32)
            sims  = embs @ q_emb

            # N4 granularity boost
            for i, c in enumerate(chunks):
                nt = c["node_type"]
                if nt == "module_summary" and cfg.granularity == "module":
                    sims[i] *= cfg.granularity_boost
                elif nt == "function" and cfg.granularity == "function":
                    sims[i] *= 1.15
                elif nt == "class" and intent == "explain":
                    sims[i] *= 1.10

            top_idx = np.argsort(-sims)[:cfg.top_k]

            # N3 session scoring
            scored = []
            for i in top_idx:
                rid   = chunks[i]["id"]
                score = float(sims[i])
                fname = chunks[i]["file"]
                if rid in session_mem._retrieved:
                    score *= session_mem.REDUNDANCY_PENALTY_LAST_TURN
                if fname in session_mem._active_files:
                    score *= (1.0 + session_mem.SESSION_ZONE_BONUS
                              * session_mem._active_files[fname])
                scored.append((rid, score))
            scored.sort(key=lambda x: -x[1])
            retrieved_ids = [rid for rid, _ in scored[:k]]

            preds.append({"query_id": qid, "retrieved": retrieved_ids,
                          "ground_truth": gt, "intent": intent})

            session_mem.record_turn(query,
                [{"file": r.split("::")[0] if "::" in r else r,
                  "node_name": r.split("::")[-1] if "::" in r else r,
                  "matched_funcs": []} for r in retrieved_ids],
                f"[{intent}] {query[:40]}")
    return preds


def run_chatgit_full(retrievers, queries, embed_model, k=10):
    """
    ChatGIT with all 5 novelties:
      N1 – git volatility weight applied to each retrieved chunk
      N2 – hybrid PageRank + query-conditioned graph attention rescoring
      N3 – session memory: redundancy penalty + session zone bonus
      N4 – intent-driven granularity boost + dynamic top_k
      N5 – call-graph neighbourhood: neighbour chunks boosted in score
    """
    by_conv = defaultdict(list)
    for row in queries:
        by_conv[row[5]].append(row)

    preds = []
    for conv_id, conv_rows in by_conv.items():
        repo_id = conv_rows[0][4]
        if repo_id not in retrievers:
            continue
        rv     = retrievers[repo_id]
        chunks = rv["chunks"]
        embs   = rv["embs"]
        git_az = rv["git_analyzer"]       # N1
        hyb_sc = rv["hybrid_scorer"]      # N2
        pagerank = rv["pagerank"]         # N5
        session_mem = SessionRetrievalMemory()

        for qid, query, gt, intent, _, _ in conv_rows:
            if not gt:
                continue
            # N4: intent classification
            cfg = classify_intent(query)
            # N3: coreference resolution
            resolved = session_mem.resolve_coreferences(query)
            q_emb = embed_model.encode([resolved], normalize_embeddings=True)[0].astype(np.float32)
            sims  = embs @ q_emb

            # N4: granularity boost
            for i, c in enumerate(chunks):
                nt = c["node_type"]
                if nt == "module_summary" and cfg.granularity == "module":
                    sims[i] *= cfg.granularity_boost
                elif nt == "function" and cfg.granularity == "function":
                    sims[i] *= 1.15
                elif nt == "class" and intent == "explain":
                    sims[i] *= 1.10

            # N2: hybrid importance rescoring
            hybrid_scores = {}
            if hyb_sc._built and hyb_sc.graph is not None:
                try:
                    hybrid_scores = hyb_sc.score_all(resolved, q_emb)
                except Exception:
                    pass

            # N1: build file-level volatility map
            recency_focused = any(kw in resolved.lower() for kw in
                                  ['recent', 'changed', 'latest', 'updated', 'new', 'modified'])

            # N5: build neighbour set from call graph
            neighbour_boost = {}   # chunk_id -> extra score
            if pagerank is not None:
                # Look at top-15 pre-N1N2 candidates to find neighbours
                pre_top = np.argsort(-sims)[:15]
                for i in pre_top:
                    c = chunks[i]
                    cname = c["node_name"]
                    fname = c["file"]
                    qname = f"{fname}::{cname}"
                    if qname not in pagerank.function_graph:
                        continue
                    succs = list(pagerank.function_graph.successors(qname))[:3]
                    preds_graph = list(pagerank.function_graph.predecessors(qname))[:3]
                    for neighbour_qname in succs + preds_graph:
                        # Find chunk index for this neighbour
                        for j, nc in enumerate(chunks):
                            nid = nc["id"]
                            if nid == neighbour_qname or nid.endswith(f"::{neighbour_qname.split('::')[-1]}"):
                                neighbour_boost[nid] = neighbour_boost.get(nid, 0) + 0.05

            # Combine scores: sim * N1_weight * (1 + N2_boost) + N5_neighbour
            top_k_pool = min(cfg.top_k * 2, len(chunks))
            top_idx = np.argsort(-sims)[:top_k_pool]

            scored = []
            for i in top_idx:
                rid   = chunks[i]["id"]
                fname = chunks[i]["file"]
                score = float(sims[i])

                # N1: git volatility weight
                n1_weight = git_az.get_retrieval_weight(fname, recency_focused)
                score *= n1_weight

                # N2: hybrid importance boost (add as additive factor)
                cname_key = f"{fname}::{chunks[i]['node_name']}"
                h = hybrid_scores.get(cname_key, 0.0)
                if h > 0:
                    score *= (1.0 + 0.3 * h)

                # N5: call-neighbourhood boost
                score += neighbour_boost.get(rid, 0.0)

                # N3: session memory
                if rid in session_mem._retrieved:
                    score *= session_mem.REDUNDANCY_PENALTY_LAST_TURN
                if fname in session_mem._active_files:
                    score *= (1.0 + session_mem.SESSION_ZONE_BONUS
                              * session_mem._active_files[fname])

                scored.append((rid, score))

            scored.sort(key=lambda x: -x[1])
            retrieved_ids = [rid for rid, _ in scored[:k]]

            preds.append({"query_id": qid, "retrieved": retrieved_ids,
                          "ground_truth": gt, "intent": intent})

            session_mem.record_turn(query,
                [{"file": r.split("::")[0] if "::" in r else r,
                  "node_name": r.split("::")[-1] if "::" in r else r,
                  "matched_funcs": []} for r in retrieved_ids],
                f"[{intent}] {query[:40]}")
    return preds


def redundancy_rate(preds, sessions):
    total = redundant = 0
    preds_map = {p["query_id"]: p["retrieved"] for p in preds}
    for sess in sessions:
        seen = set()
        for turn in sess["turns"]:
            for cid in preds_map.get(turn["qid"], []):
                total += 1
                if cid in seen:
                    redundant += 1
            seen.update(preds_map.get(turn["qid"], []))
    return redundant / total if total else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  ChatGIT × ConvCodeBench — Real Evaluation")
    print(f"  Repos: {list(REPOS.keys())}")
    print("=" * 70)

    print("\n[1/4] Chunking repositories...")
    all_chunks = chunk_all_repos()

    print("\n[2/4] Loading ConvCodeBench conversations...")
    queries, sessions = load_conversations(all_chunks)
    queries_gt = [q for q in queries if q[2]]   # only turns with matched GT
    print(f"  Total turns: {len(queries)} | With GT: {len(queries_gt)} "
          f"| Conversations: {len(sessions)}")
    ic = defaultdict(int)
    for *_, intent, repo_id, _ in queries_gt:
        ic[intent] += 1
    print(f"  Intent dist: {dict(sorted(ic.items()))}")

    print("\n[3/4] Building retrievers + embeddings (BGE-small)...")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5",
                                       cache_folder="/tmp/hf_cache")
    retrievers = build_retrievers(all_chunks, embed_model)

    print("\n[4/4] Running retrieval systems...")
    k = 10
    t0 = time.time()
    bm25_preds      = run_lexical("bm25",      retrievers, queries_gt, k)
    bm25t_preds     = run_lexical("bm25_tuned",retrievers, queries_gt, k)
    vanilla_preds   = run_dense(retrievers, queries_gt, embed_model, k)
    chatgit_preds   = run_chatgit(retrievers, queries_gt, embed_model, k)
    print(f"  All systems done in {time.time()-t0:.1f}s")

    systems = {"BM25":              bm25_preds,
               "BM25-Tuned(RepoCoder proxy)": bm25t_preds,
               "VanillaRAG(BGE)":  vanilla_preds,
               "ChatGIT(N3+N4+BGE)": chatgit_preds}

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  REAL RESULTS — ConvCodeBench ({len(queries_gt)} queries, "
          f"{len(sessions)} conversations, {len(REPOS)} repos)")
    print("=" * 70)

    results = {}
    for name, preds in systems.items():
        res = evaluate_retrieval(preds, ks=[1,5,10], n_bootstrap=3000)
        results[name] = res
        print_retrieval_report(res, title=name)

    # ── Main table ────────────────────────────────────────────────────────────
    metrics = ["mrr","recall@5","ndcg@5","p@1","success@5","success@10"]
    print("\n" + "=" * 70)
    print("  MAIN RESULTS TABLE")
    print("=" * 70)
    print(f"  {'System':<22}" + "".join(f"  {m:>12}" for m in metrics))
    print("  " + "-" * 100)
    for name, res in results.items():
        row = f"  {name:<22}"
        for m in metrics:
            s = res["summary"].get(m, {})
            v = s.get("mean", 0.0)
            ci = (s.get("ci_hi", v) - s.get("ci_lo", v)) / 2
            row += f"  {v:>6.4f}±{ci:>4.4f}"
        print(row)

    # ── Per-intent ────────────────────────────────────────────────────────────
    intents = ["locate","explain","summarize","debug"]
    print(f"\n  PER-INTENT MRR")
    print(f"  {'System':<22}" + "".join(f"  {i:>12}" for i in intents))
    print("  " + "-" * 75)
    for name, res in results.items():
        pi  = res.get("per_intent", {})
        row = f"  {name:<22}"
        for intent in intents:
            v = pi.get(intent, {}).get("mrr", 0.0)
            row += f"  {v:>12.4f}"
        print(row)

    # ── Per-repo ──────────────────────────────────────────────────────────────
    print(f"\n  PER-REPO MRR  (ChatGIT vs BM25)")
    print(f"  {'Repo':<14}  {'ChatGIT':>10}  {'BM25':>8}  {'VanillaRAG':>12}  {'Delta CG-BM':>12}")
    print("  " + "-" * 62)
    per_repo = defaultdict(lambda: defaultdict(list))
    for name, preds in systems.items():
        for p in preds:
            repo = "_".join(p["query_id"].split("_conv_")[0].split("_"))
            mrr  = next((1/i for i, c in enumerate(p["retrieved"],1)
                         if c in p["ground_truth"]), 0.0)
            per_repo[repo][name].append(mrr)
    for repo in sorted(per_repo):
        cg = np.mean(per_repo[repo].get("ChatGIT(N3+N4+BGE)",[0]))
        bm = np.mean(per_repo[repo].get("BM25",[0]))
        vr = np.mean(per_repo[repo].get("VanillaRAG(BGE)",[0]))
        print(f"  {repo:<14}  {cg:>10.4f}  {bm:>8.4f}  {vr:>12.4f}  {cg-bm:>+12.4f}")

    # ── Redundancy rate (N3) ──────────────────────────────────────────────────
    print(f"\n  REDUNDANCY RATE (↓ better — N3 effectiveness)")
    for name, preds in systems.items():
        rr = redundancy_rate(preds, sessions)
        print(f"  {name:<22}  {rr:.4f}")

    # ── Statistical significance ───────────────────────────────────────────────
    print(f"\n  STATISTICAL SIGNIFICANCE (ChatGIT vs baselines, n={len(queries_gt)})")
    cg_q   = {p["query_id"]: p for p in results["ChatGIT(N3+N4+BGE)"]["per_query"]}
    reports = []
    for name in ["BM25","BM25-Tuned(RepoCoder proxy)","VanillaRAG(BGE)"]:
        oth_q = {p["query_id"]: p for p in results[name]["per_query"]}
        common = sorted(set(cg_q) & set(oth_q))
        if len(common) < 2:
            continue
        for metric in ["mrr","recall@5","ndcg@5"]:
            a = np.array([cg_q[q][metric]  for q in common])
            b = np.array([oth_q[q][metric] for q in common])
            reports.append(full_comparison_report(
                a, b, metric_name=f"{metric} vs {name}",
                system_a_name="ChatGIT", system_b_name=name))
    print_comparison_table(reports)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    save = {}
    for name, res in results.items():
        save[name] = {m: {"mean":  res["summary"].get(m,{}).get("mean",0),
                           "ci_lo": res["summary"].get(m,{}).get("ci_lo",0),
                           "ci_hi": res["summary"].get(m,{}).get("ci_hi",0)}
                      for m in metrics}
        save[name]["per_intent"] = {
            intent: {"mrr": res["per_intent"].get(intent,{}).get("mrr",0)}
            for intent in intents}
        save[name]["redundancy_rate"] = redundancy_rate(
            systems[name], sessions)
    with open("results/convcodebench_results.json", "w") as f:
        json.dump(save, f, indent=2)
    print("\n  Saved → results/convcodebench_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
