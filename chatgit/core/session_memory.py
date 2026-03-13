"""
Multi-Turn Session-Aware Retrieval Memory — Novelty 3

Maintains per-session retrieval state across conversation turns to enable:

1. Redundancy suppression
   Chunks retrieved recently get a score penalty — avoids showing the same
   code block across multiple turns.

2. Session-zone coherence bonus
   Files active in recent turns receive a mild retrieval boost so follow-up
   questions naturally continue exploring the same topic area.

3. Discussed-function bonus
   Functions explicitly mentioned in prior responses get a boost when they
   appear in candidates for a new query, supporting drill-down conversations.

4. Code co-reference resolution
   Short queries with pronouns ("what does it return?", "the other one") are
   expanded with the last-discussed function / active file as a context hint,
   improving retrieval accuracy on follow-up turns.
"""

import re
from collections import defaultdict
from typing import Dict, List


class SessionRetrievalMemory:
    """
    Retrieval memory for a single chat session.
    Reset by calling .reset() when a new repository is loaded.
    """

    REDUNDANCY_PENALTY_SAME_TURN   = 0.30   # chunk retrieved this turn
    REDUNDANCY_PENALTY_LAST_TURN   = 0.60   # chunk retrieved 1 turn ago
    REDUNDANCY_PENALTY_OLDER       = 0.85   # 2+ turns ago
    SESSION_ZONE_BONUS             = 0.30   # additive bonus fraction per unit weight
    DISCUSSED_FUNC_BONUS           = 1.20   # multiplier for discussed functions
    RECENCY_DECAY                  = 0.80   # per-turn decay on active-file weights

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        self.turn: int = 0
        # chunk_id -> list of turn numbers it was retrieved
        self._retrieved: Dict[str, List[int]] = defaultdict(list)
        # file -> current zone weight (decays each turn)
        self._active_files: Dict[str, float] = {}
        # functions discussed across the session (most recent last)
        self._discussed_fns: List[str] = []

    # ------------------------------------------------------------------
    # Record a completed turn
    # ------------------------------------------------------------------

    def record_turn(self, query: str, retrieved_chunks: List[dict], response: str):
        """
        Call after the LLM response is generated to update memory.

        retrieved_chunks: list of dicts with keys:
            'file'         : str  (file_name metadata)
            'node_name'    : str  (function/class name)
            'matched_funcs': list[str]
        """
        self.turn += 1

        # Decay existing active files
        decayed = {f: w * self.RECENCY_DECAY for f, w in self._active_files.items()}
        self._active_files = {f: w for f, w in decayed.items() if w > 0.05}

        for chunk in retrieved_chunks:
            fname     = chunk.get("file", "")
            node_name = chunk.get("node_name", "")
            chunk_id  = f"{fname}::{node_name}"

            self._retrieved[chunk_id].append(self.turn)
            # Fresh weight for this file
            self._active_files[fname] = max(self._active_files.get(fname, 0.0), 1.0)

            for fn in chunk.get("matched_funcs", []):
                if fn and fn not in self._discussed_fns:
                    self._discussed_fns.append(fn)

        # Keep discussed functions list bounded
        self._discussed_fns = self._discussed_fns[-20:]

    # ------------------------------------------------------------------
    # Apply session scores to candidates
    # ------------------------------------------------------------------

    def apply_session_scores(self, candidates: List[dict]) -> List[dict]:
        """
        Adjust candidate scores in-place based on session memory.
        Should be called AFTER the initial vector + hybrid scoring,
        BEFORE cross-encoder reranking.
        """
        if self.turn == 0:
            return candidates  # no history yet

        for c in candidates:
            meta      = c["snippet"].metadata
            fname     = meta.get("file_name", "")
            node_name = meta.get("node_name", "")
            chunk_id  = f"{fname}::{node_name}"

            # --- Redundancy penalty ---
            if chunk_id in self._retrieved:
                last_seen  = max(self._retrieved[chunk_id])
                turns_ago  = self.turn - last_seen
                if turns_ago == 0:
                    c["score"] *= self.REDUNDANCY_PENALTY_SAME_TURN
                elif turns_ago == 1:
                    c["score"] *= self.REDUNDANCY_PENALTY_LAST_TURN
                else:
                    c["score"] *= self.REDUNDANCY_PENALTY_OLDER

            # --- Session-zone bonus ---
            if fname in self._active_files:
                c["score"] *= (1.0 + self.SESSION_ZONE_BONUS * self._active_files[fname])

            # --- Discussed-function bonus ---
            for fn in c.get("matched_funcs", []):
                if fn in self._discussed_fns:
                    c["score"] *= self.DISCUSSED_FUNC_BONUS
                    break

        return candidates

    # ------------------------------------------------------------------
    # Co-reference resolution
    # ------------------------------------------------------------------

    _PRONOUN_PATTERNS = [
        r"\bit\b",
        r"\bthis function\b",
        r"\bthe other one\b",
        r"\bthe same\b",
        r"\bthat function\b",
        r"\bthis class\b",
        r"\bthe above\b",
    ]

    def resolve_coreferences(self, query: str) -> str:
        """
        Expand short pronoun-heavy queries with session context.
        E.g. "what does it return?" → "what does <last_func> return?"
        """
        if not self._discussed_fns and not self._active_files:
            return query

        q = query.strip()
        q_lower = q.lower()

        # Only attempt resolution for short queries that contain pronouns
        if len(q.split()) < 10:
            for pattern in self._PRONOUN_PATTERNS:
                if re.search(pattern, q_lower):
                    if self._discussed_fns:
                        last_fn = self._discussed_fns[-1]
                        q = f"{q} [context: function '{last_fn}']"
                    break

        # If query has no clear subject at all, hint with the most active file
        if len(q.split()) < 5 and self._active_files:
            top_file = max(self._active_files, key=self._active_files.get)
            q = f"{q} [file context: {top_file}]"

        return q

    # ------------------------------------------------------------------
    # Session summary for the prompt
    # ------------------------------------------------------------------

    def get_session_summary(self) -> str:
        """
        One-paragraph summary of session state to include in the LLM prompt.
        Helps the model give coherent follow-up answers.
        """
        if self.turn == 0:
            return ""

        lines = ["\n## Session Context (Multi-Turn Memory)"]
        if self._discussed_fns:
            fns = ", ".join(f"`{f}`" for f in self._discussed_fns[-5:])
            lines.append(f"- Recently discussed functions: {fns}")
        if self._active_files:
            top = sorted(self._active_files.items(), key=lambda x: x[1], reverse=True)[:3]
            files = ", ".join(f"`{f}`" for f, _ in top)
            lines.append(f"- Active topic files: {files}")
        lines.append(f"- Conversation turn: {self.turn}")
        return "\n".join(lines)
