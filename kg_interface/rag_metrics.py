"""Required RAG metrics report for dense and LiteRag retrieval."""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np

from litrag import KnowledgeGraph, LogisticPolicy, RetrievalEnv
from litrag.agent import _seeds
from litrag.evaluate import ALWAYS_EXPAND, NEVER_EXPAND
from litrag.retrieve import rollout


WORD = re.compile(r"[a-z0-9]+")


def normalize(text):
    return " ".join(WORD.findall(str(text).lower()))


def dcg(relevances):
    return sum(
        relevance / np.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


def ranked_metrics(ranked_chunk_ids, gold_chunk_ids, k):
    gold = set(gold_chunk_ids)
    ranked = ranked_chunk_ids[:k]
    hits = [int(chunk_id in gold) for chunk_id in ranked]

    recall = sum(hits) / len(gold) if gold else 0.0
    precision = sum(hits) / k if k else 0.0
    hit_rank = next((rank + 1 for rank, hit in enumerate(hits) if hit), None)
    mrr = 1.0 / hit_rank if hit_rank else 0.0
    ndcg = (
        dcg(hits) /
        dcg([1] * min(len(gold), k))
        if gold else 0.0
    )
    return recall, precision, mrr, ndcg


def answer_in_text(answer, chunk_texts):
    answer_norm = normalize(answer)
    if not answer_norm or answer_norm in {"yes", "no"}:
        return answer_norm in {
            normalize(text)
            for text in chunk_texts
        }
    return any(answer_norm in normalize(text) for text in chunk_texts)


def dense_retrieve(kg, question, k):
    query = kg.encode_query(question)
    ranked = np.argsort(-(kg.chunk_emb @ query))[:k].tolist()
    return ranked


def graph_retrieve(kg, env, qa, choose_action, k, link):
    seeds = _seeds(kg, qa, link)
    started = time.perf_counter()
    rollout(env, qa, seeds, choose_action)
    latency = time.perf_counter() - started
    ranked = env.top_chunks(k)
    return ranked, env.steps, latency


def summarize(kg, qa_list, ranked_results, k):
    recalls, precisions, mrrs, ndcgs = [], [], [], []
    answer_present = []
    for qa, ranked in zip(qa_list, ranked_results):
        gold_chunks = [
            kg.chunk_idx[chunk_id]
            for chunk_id in qa.get("supporting_chunk_ids", [])
            if chunk_id in kg.chunk_idx
        ]
        recall, precision, mrr, ndcg = ranked_metrics(ranked, gold_chunks, k)
        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)
        ndcgs.append(ndcg)

        texts = [kg.chunks[chunk_id]["text"] for chunk_id in ranked]
        answer_present.append(float(answer_in_text(qa.get("answer", ""), texts)))

        gold_entities = {
            kg.ent_idx[entity_id]
            for entity_id in qa.get("answer_entity_ids", [])
            if entity_id in kg.ent_idx
        }
    return {
        f"support_recall@{k}": float(np.mean(recalls)),
        f"support_precision@{k}": float(np.mean(precisions)),
        f"mrr@{k}": float(np.mean(mrrs)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "answer_in_top_chunks": float(np.mean(answer_present)),
    }


def evaluate_system(kg, qa_list, k, policy=None, params=None):
    dense_ranked, dense_latencies = [], []
    for qa in qa_list:
        started = time.perf_counter()
        dense_ranked.append(dense_retrieve(kg, qa["question"], k))
        dense_latencies.append(time.perf_counter() - started)
    dense = summarize(kg, qa_list, dense_ranked, k)
    dense.update({
        "avg_latency_seconds": float(np.mean(dense_latencies)),
    })

    env = RetrievalEnv(
        kg,
        max_steps=params["max_steps"] if params else 6,
        step_cost=params["step_cost"] if params else 0.05,
    )
    link = params["link"] if params else False

    modes = {
        "never_expand": NEVER_EXPAND,
        "always_expand": ALWAYS_EXPAND,
    }
    if policy is not None:
        modes["learned_rl"] = lambda features: policy.act(features, greedy=True)

    results = {"dense": dense}
    for mode_name, choose_action in modes.items():
        ranked_results, steps, latencies = [], [], []
        answer_entity_hits = []
        for qa in qa_list:
            ranked, step_count, latency = graph_retrieve(
                kg, env, qa, choose_action, k, link
            )
            ranked_results.append(ranked)
            steps.append(step_count)
            latencies.append(latency)

            gold_entities = {
                kg.ent_idx[entity_id]
                for entity_id in qa.get("answer_entity_ids", [])
                if entity_id in kg.ent_idx
            }
            answer_entity_hits.append(
                float(bool(gold_entities & env.depth.keys()))
            )

        metrics = summarize(kg, qa_list, ranked_results, k)
        metrics.update({
            "answer_in_graph": float(np.mean(answer_entity_hits)),
            "avg_expansion_steps": float(np.mean(steps)),
            "avg_latency_seconds": float(np.mean(latencies)),
        })
        results[mode_name] = metrics

    efficiency = []
    for mode_name, metrics in results.items():
        recall = metrics[f"support_recall@{k}"]
        answer = metrics.get("answer_in_top_chunks", metrics.get("answer_in_graph", 0.0))
        steps = metrics.get("avg_expansion_steps", 0.0)
        efficiency.append({
            "mode": mode_name,
            "score": recall + answer - 0.02 * steps,
        })
    for mode in efficiency:
        results.setdefault("efficiency", {})[mode["mode"]] = round(mode["score"], 6)

    return results


def write_reports(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rag_metrics.json"
    csv_path = output_dir / "rag_metrics.csv"

    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["mode", *sorted({
            key
            for metrics in results.items()
            if isinstance(metrics[1], dict)
            for key in (
                metrics[1].keys()
                if metrics[0] != "efficiency"
                else []
            )
        })]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode, metrics in results.items():
            if mode == "efficiency":
                continue
            writer.writerow({"mode": mode, **metrics})

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", default="metrics")
    args = parser.parse_args()

    kg = KnowledgeGraph(args.data)
    qa_list = [qa for qa in kg.qa if qa.get("split", "train") == args.split]
    if not qa_list:
        qa_list = kg.qa
    if args.limit:
        qa_list = qa_list[:args.limit]

    params = None
    if args.params:
        params = json.loads(Path(args.params).read_text(encoding="utf-8"))

    policy = LogisticPolicy.load(args.policy) if args.policy else None
    results = evaluate_system(kg, qa_list, args.k, policy, params)
    json_path, csv_path = write_reports(results, Path(args.output))

    print(json.dumps(results, indent=2))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
