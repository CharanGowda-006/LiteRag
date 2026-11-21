"""Run episodes and the end-to-end retrieve() used at inference time."""
import numpy as np


def rollout(env, qa, seeds, choose_action):
    """qa: dict with 'question' (+ optional gold keys). Returns list of (features, action, reward)."""
    f = env.reset(qa["question"], seeds, qa.get("answer_entity_ids"), qa.get("supporting_chunk_ids"))
    traj, done = [], False
    while not done:
        a = choose_action(f)
        f2, r, done, _ = env.step(a)
        traj.append((f, a, r))
        f = f2
    return traj


def retrieve(kg, env, policy, question, k=3):
    """Question -> link entities -> RL expansion -> top-k supporting chunks (feed these to the answer generator)."""
    seeds = kg.link_entities(question)
    traj = rollout(env, {"question": question}, seeds, lambda f: policy.act(f, greedy=True))
    chunks = env.top_chunks(k)
    return {
        "seed_entities": [kg.ent_names[s] for s in seeds],
        "steps": env.steps,
        "visited_entities": [kg.ent_names[e] for e in env.depth],
        "chunks": [{"chunk_id": kg.chunk_ids[c], "text": kg.chunks[c]["text"], "score": float(env.chunk_sim[c])} for c in chunks],
    }
