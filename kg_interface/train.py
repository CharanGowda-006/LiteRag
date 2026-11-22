"""Train + evaluate the RL retrieval policy.   python train.py --data data --epochs 15"""
import argparse, json
from litrag import KnowledgeGraph, RetrievalEnv, LogisticPolicy, train_policy, retrieve
from litrag.evaluate import evaluate, dense_baseline, ALWAYS_EXPAND, NEVER_EXPAND
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--max-steps", type=int, default=6)
ap.add_argument("--step-cost", type=float, default=0.05)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--link", action="store_true", help="use entity linking instead of gold seed entities")
ap.add_argument("--out", default="policy.npz")
ap.add_argument("--search-params", action="store_true",
                help="compare efficient max-step and step-cost configurations")
ap.add_argument("--train-limit", type=int, default=None)
ap.add_argument("--dev-limit", type=int, default=None)
a = ap.parse_args()

kg = KnowledgeGraph(a.data)
env = RetrievalEnv(kg, a.max_steps, a.step_cost)
split = lambda s: [q for q in kg.qa if q.get("split", "train") == s]
train, dev, test = split("train"), split("dev"), split("test")
train = train[:a.train_limit] if a.train_limit else train
dev = dev[:a.dev_limit] if a.dev_limit else dev
print(f"{len(kg.ent_ids)} entities, {len(kg.chunks)} chunks | QA train/dev/test = {len(train)}/{len(dev)}/{len(test)}")

if a.search_params:
    candidate_steps = (2, 4, 6, 8)
    candidate_costs = (0.01, 0.05, 0.10)
    search_set = dev or train
    best = None
    for max_steps in candidate_steps:
        for step_cost in candidate_costs:
            candidate_env = RetrievalEnv(kg, max_steps=max_steps, step_cost=step_cost)
            candidate_policy = train_policy(
                kg, candidate_env, train, epochs=a.epochs, link=a.link, verbose=False
            )
            metrics = evaluate(
                kg, candidate_env, search_set,
                lambda f, p=candidate_policy: p.act(f, greedy=True),
                a.k, a.link,
            )
            score = (metrics[f"support_recall@{a.k}"] +
                     metrics["answer_in_graph"] - 0.02 * metrics["avg_steps"])
            if best is None or score > best["score"]:
                best = {"score": score, "max_steps": max_steps,
                        "step_cost": step_cost, "policy": candidate_policy,
                        "metrics": metrics}
    a.max_steps = best["max_steps"]
    env = RetrievalEnv(kg, a.max_steps, best["step_cost"])
    policy = train_policy(kg, env, train, epochs=a.epochs, link=a.link)
    print("Selected efficient parameters:",
          json.dumps({"max_steps": a.max_steps, "step_cost": best["step_cost"],
                      "dev_metrics": best["metrics"], "objective": round(best["score"], 4)}))
else:
    policy = train_policy(kg, env, train, epochs=a.epochs, link=a.link)
policy.save(a.out)
params_path = Path(a.out).with_name("rl_params.json")
params_path.write_text(json.dumps({
    "max_steps": a.max_steps,
    "step_cost": env.step_cost,
    "k": a.k,
    "link": a.link,
}, indent=2), encoding="utf-8")

eval_set = test or dev or train
print(f"\nEvaluation on {len(eval_set)} questions (k={a.k})")
rows = {
    "dense only (no graph)": dense_baseline(kg, eval_set, a.k),
    "never expand": evaluate(kg, env, eval_set, NEVER_EXPAND, a.k, a.link),
    "always expand": evaluate(kg, env, eval_set, ALWAYS_EXPAND, a.k, a.link),
    "learned RL policy": evaluate(kg, env, eval_set, lambda f: policy.act(f, greedy=True), a.k, a.link),
}
for name, m in rows.items():
    print(f"{name:24s}", {k: round(float(v), 3) for k, v in m.items()})

print("\nDemo:", json.dumps(retrieve(kg, env, policy, eval_set[0]["question"], a.k), indent=1))
