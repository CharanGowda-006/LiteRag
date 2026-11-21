"""Metrics + baselines (dense-only, never-expand, always-expand) to compare against the learned policy."""
import numpy as np
from .env import EXPAND, STOP
from .retrieve import rollout
from .agent import _seeds


def evaluate(kg, env, qa_list, choose_action, k=3, link=False):
    hit, rec, steps = [], [], []
    for qa in qa_list:
        rollout(env, qa, _seeds(kg, qa, link), choose_action)
        gold_ans = {kg.ent_idx[a] for a in qa.get("answer_entity_ids", [])}
        gold_sup = {kg.chunk_idx[c] for c in qa.get("supporting_chunk_ids", [])}
        hit.append(float(bool(gold_ans & env.depth.keys())))
        top = set(env.top_chunks(k))
        rec.append(len(gold_sup & top) / len(gold_sup) if gold_sup else 0.0)
        steps.append(env.steps)
    return {"answer_in_graph": np.mean(hit), f"support_recall@{k}": np.mean(rec), "avg_steps": np.mean(steps)}


def dense_baseline(kg, qa_list, k=3):
    """No graph at all: top-k chunks by embedding similarity over the whole corpus."""
    rec = []
    for qa in qa_list:
        top = set(np.argsort(-(kg.chunk_emb @ kg.encode_query(qa["question"])))[:k].tolist())
        gold = {kg.chunk_idx[c] for c in qa.get("supporting_chunk_ids", [])}
        rec.append(len(gold & top) / len(gold) if gold else 0.0)
    return {f"support_recall@{k}": np.mean(rec)}


ALWAYS_EXPAND = lambda f: EXPAND
NEVER_EXPAND = lambda f: STOP
