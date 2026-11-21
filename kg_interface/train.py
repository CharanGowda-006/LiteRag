"""Train + evaluate the RL retrieval policy.   python train.py --data data --epochs 15"""
import argparse, json
from litrag import KnowledgeGraph, RetrievalEnv, LogisticPolicy, train_policy, retrieve
from litrag.evaluate import evaluate, dense_baseline, ALWAYS_EXPAND, NEVER_EXPAND

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--max-steps", type=int, default=6)
ap.add_argument("--step-cost", type=float, default=0.05)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--link", action="store_true", help="use entity linking instead of gold seed entities")
ap.add_argument("--out", default="policy.npz")
a = ap.parse_args()

kg = KnowledgeGraph(a.data)
env = RetrievalEnv(kg, a.max_steps, a.step_cost)
split = lambda s: [q for q in kg.qa if q.get("split", "train") == s]
train, dev, test = split("train"), split("dev"), split("test")
print(f"{len(kg.ent_ids)} entities, {len(kg.chunks)} chunks | QA train/dev/test = {len(train)}/{len(dev)}/{len(test)}")

policy = train_policy(kg, env, train, epochs=a.epochs, link=a.link)
policy.save(a.out)

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
