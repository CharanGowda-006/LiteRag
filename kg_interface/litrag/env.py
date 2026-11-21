"""RL environment: start from seed entities, repeatedly choose EXPAND or STOP."""
import math
import numpy as np

STOP, EXPAND = 0, 1
N_FEATURES = 10


class RetrievalEnv:
    """
    State  : visited entities (with depth), frontier (visited but not yet expanded), chunks attached to visited entities.
    Action : EXPAND -> expand the frontier node most similar to the query (its neighbours join visited/frontier)
             STOP   -> end the episode and return the chunks collected so far.
    Reward : (gain in retrieval score) - step_cost per EXPAND. Score uses gold labels (answer entity + supporting chunks),
             which are used ONLY for the reward, never for the features.
    """

    def __init__(self, kg, max_steps=6, step_cost=0.05):
        self.kg, self.max_steps, self.step_cost = kg, max_steps, step_cost

    # ---------- episode control ----------
    def reset(self, question, seeds, answer_ids=None, support_ids=None):
        kg = self.kg
        q = kg.encode_query(question)
        self.ent_sim, self.chunk_sim = kg.ent_emb @ q, kg.chunk_emb @ q
        self.depth, self.frontier, self.chunks = {}, set(), set()
        for s in seeds:
            self._add(s, 0)
        self.steps = 0
        self.gold_ans = {kg.ent_idx[a] for a in (answer_ids or [])}
        self.gold_sup = {kg.chunk_idx[c] for c in (support_ids or [])}
        self.prev_best = self._best_chunk()
        return self._features()

    def step(self, action):
        before = self._score()
        self.prev_best = self._best_chunk()
        if action == STOP or not self.frontier:
            return self._features(), 0.0, True, {}
        node = self._best_frontier()
        self.frontier.discard(node)
        for nb in self.kg.nbrs[node]:
            if nb not in self.depth:
                self._add(nb, self.depth[node] + 1)
        self.steps += 1
        reward = self._score() - before - self.step_cost
        done = self.steps >= self.max_steps or not self.frontier
        return self._features(), reward, done, {}

    # ---------- helpers ----------
    def _add(self, e, d):
        self.depth[e] = d
        self.frontier.add(e)
        self.chunks.update(self.kg.entity_chunks.get(e, []))

    def _best_frontier(self):
        return max(self.frontier, key=lambda e: (self.ent_sim[e], -self.depth[e]))

    def _best_chunk(self):
        return float(self.chunk_sim[list(self.chunks)].max()) if self.chunks else 0.0

    def _score(self):
        if not self.gold_ans and not self.gold_sup:
            return 0.0
        ans = 1.0 if self.gold_ans & self.depth.keys() else 0.0
        sup = len(self.gold_sup & self.chunks) / len(self.gold_sup) if self.gold_sup else 0.0
        return 0.5 * ans + 0.5 * sup

    def _features(self):
        if self.frontier:
            fr = list(self.frontier)
            best = self._best_frontier()
            best_sim, mean_sim = float(self.ent_sim[best]), float(self.ent_sim[fr].mean())
            best_depth = self.depth[best] / self.max_steps
            best_deg = math.log1p(len(self.kg.nbrs[best])) / 5
        else:
            best_sim = mean_sim = best_depth = best_deg = 0.0
        cur = self._best_chunk()
        return np.array([
            self.steps / self.max_steps,           # 0 progress
            math.log1p(len(self.frontier)) / 5,    # 1 frontier size
            math.log1p(len(self.depth)) / 5,       # 2 visited size
            best_sim,                              # 3 best frontier node similarity to query
            mean_sim,                              # 4 mean frontier similarity
            cur,                                   # 5 best collected chunk similarity
            cur - self.prev_best,                  # 6 improvement from last step
            best_depth,                            # 7 depth of best frontier node
            best_deg,                              # 8 degree of best frontier node
            math.log1p(len(self.chunks)) / 5,      # 9 chunks collected
        ], dtype=np.float32)

    def top_chunks(self, k=3):
        return sorted(self.chunks, key=lambda c: -self.chunk_sim[c])[:k]
