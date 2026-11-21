"""Logistic (Bernoulli) policy trained with REINFORCE + moving-average baseline. Pure numpy."""
import numpy as np
from .env import EXPAND, STOP, N_FEATURES
from .retrieve import rollout


class LogisticPolicy:
    def __init__(self, n_features=N_FEATURES, lr=0.05):
        self.w, self.b, self.lr = np.zeros(n_features), 1.0, lr  # b>0: starts biased towards EXPAND

    def p_expand(self, f):
        return 1.0 / (1.0 + np.exp(-(self.w @ f + self.b)))

    def act(self, f, rng=None, greedy=False):
        p = self.p_expand(f)
        if greedy or rng is None:
            return EXPAND if p >= 0.5 else STOP
        return EXPAND if rng.random() < p else STOP

    def update(self, traj, baseline, gamma=1.0):
        G = 0.0
        for f, a, r in reversed(traj):
            G = r + gamma * G
            grad = (a - self.p_expand(f)) * (G - baseline)   # d log pi / d logit * advantage
            self.w += self.lr * grad * f
            self.b += self.lr * grad
        return G

    def save(self, path):
        np.savez(path, w=self.w, b=self.b)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        p = cls(len(d["w"]))
        p.w, p.b = d["w"], float(d["b"])
        return p


def _seeds(kg, qa, link):
    if link or not qa.get("entities"):
        return kg.link_entities(qa["question"])
    return [kg.ent_idx[e] for e in qa["entities"]]


def train_policy(kg, env, qa_list, policy=None, epochs=10, seed=0, link=False, verbose=True):
    policy, rng, baseline = policy or LogisticPolicy(), np.random.default_rng(seed), 0.0
    for ep in range(epochs):
        order, rets = rng.permutation(len(qa_list)), []
        for i in order:
            qa = qa_list[i]
            traj = rollout(env, qa, _seeds(kg, qa, link), lambda f: policy.act(f, rng))
            if not traj:
                continue
            G = policy.update(traj, baseline)
            baseline += 0.01 * (G - baseline)
            rets.append(G)
        if verbose:
            print(f"epoch {ep + 1}/{epochs}  mean return {np.mean(rets):.3f}  w[:4]={np.round(policy.w[:4], 2)} b={policy.b:.2f}")
    return policy
