# LiteRag RL: formulas and implementation

This note explains the reinforcement-learning policy used to decide whether
LiteRag should expand its knowledge graph or stop retrieval. The formulas and
behavior below describe the current implementation in `litrag/env.py` and
`litrag/agent.py`.

## 1. Retrieval state and action

An episode starts with seed entities linked from the question or supplied by
the QA record. Every visited entity contributes its attached chunks to the
retrieval set.

The policy receives a 10-element state vector \(f_t\):

| Index | Feature | Implementation |
|---:|---|---|
| 0 | Progress: `steps / max_steps` | Normalized expansion count |
| 1 | `log1p(frontier_size) / 5` | Number of visited entities still available to expand |
| 2 | `log1p(visited_entity_count) / 5` | Number of visited entities |
| 3 | Query similarity of the best frontier entity | Entity with the highest query similarity |
| 4 | Mean query similarity of frontier entities | Current frontier's mean similarity |
| 5 | Best query similarity among collected chunks | Current retrieval-set quality |
| 6 | Change in best chunk similarity since the previous state | Retrieval improvement from the last expansion |
| 7 | Best frontier entity's depth divided by `max_steps` | Normalized graph depth |
| 8 | `log1p(best_frontier_degree) / 5` | Degree of the best frontier entity |
| 9 | `log1p(collected_chunk_count) / 5` | Number of chunks currently collected |

If the frontier is empty, features that require a best frontier entity are set
to zero. Feature construction is in `RetrievalEnv._features` in `litrag/env.py`.

The action is binary:

- \(a_t=1\): `EXPAND`
- \(a_t=0\): `STOP`

The policy chooses only whether to expand. It does not choose a specific
frontier entity. On `EXPAND`, the environment expands the frontier entity with
the highest query similarity, adds unvisited neighbors, and collects their
chunks. A rollout also stops when the frontier is empty or `max_steps` is
reached. Returned chunks are ranked by query similarity.

## 2. Logistic action policy

The probability of expanding at state \(f_t\) is:

\[
z_t = w^\top f_t + b
\]
\[
p_t = P(a_t=1\mid f_t)=\sigma(z_t)
    = \frac{1}{1+\exp(-z_t)}
\]

During training, the action is sampled as a Bernoulli variable:

\[
a_t \sim \operatorname{Bernoulli}(p_t)
\]

At inference with greedy action selection:

\[
a_t =
\begin{cases}
1 & p_t \ge 0.5\\
0 & p_t < 0.5
\end{cases}
\]

The implementation is `LogisticPolicy.p_expand` and `LogisticPolicy.act` in
`litrag/agent.py`. The initial weights are \(w=\mathbf{0}\), and the initial
bias is \(b=1\); therefore the initial probability is
\(\sigma(1)\approx0.731\). The default learning rate is \(0.05\).

## 3. Retrieval score and step reward

Let \(A\) be the gold answer-entity set and \(S\) the gold supporting-chunk set.
Let \(V_t\) be visited entities and \(C_t\) collected chunks at time \(t\).

Answer-entity coverage is binary:

\[
H_t =
\begin{cases}
1 & A\cap V_t \ne \varnothing\\
0 & \text{otherwise}
\end{cases}
\]

Supporting-chunk coverage is recall:

\[
R_t =
\begin{cases}
\frac{|S\cap C_t|}{|S|} & |S|>0\\
0 & |S|=0
\end{cases}
\]

The environment score is:

\[
Q_t = 0.5H_t + 0.5R_t
\]

For an expansion, the reward is:

\[
r_t = Q_{t+1}-Q_t-c
\]

where \(c\) is `step_cost` (default `0.05`). Thus expansion gets a positive
reward only when its coverage gain exceeds the step cost. If expansion finds
nothing new, reward is \(-c\). STOP gives zero reward and ends the episode.
If both gold sets are empty, the score is zero.

The two score weights are fixed at `0.5` each. If one gold set is empty, its
term contributes zero; the remaining term is not renormalized. The implementation
is `RetrievalEnv._score` and `RetrievalEnv.step` in `litrag/env.py`.

## 4. Return, baseline, and REINFORCE update

The discount factor defaults to \(\gamma=1\). For a trajectory of length \(T\),
the return from step \(t\) is:

\[
G_t = \sum_{j=t}^{T-1}\gamma^{j-t}r_j
\]

It is computed backwards in the code as:

\[
G \leftarrow r_t + \gamma G
\]

The baseline \(B\) is a moving average of episode returns. It starts at zero
and is updated once after each episode:

\[
B \leftarrow B + 0.01(G_0-B)
\]

For each recorded state/action pair, the advantage is:

\[
\hat{A}_t = G_t-B
\]

For a Bernoulli logistic policy, the score-function derivative with respect to
the logit is \(a_t-p_t\). The implementation's policy-gradient ascent step is:

\[
\delta_t=(a_t-p_t)(G_t-B)
\]
\[
w \leftarrow w+\alpha\delta_t f_t
\]
\[
b \leftarrow b+\alpha\delta_t
\]

Here \(\alpha\) is the learning rate. The update is applied to trajectory steps
in reverse order. There is no gradient clipping, entropy bonus, critic network,
or minibatch optimizer in the current implementation. See
`LogisticPolicy.update` and `train_policy` in `litrag/agent.py`.

Training shuffles the QA order each epoch using a seeded NumPy random generator.
It samples actions during training and uses greedy actions for the learned
policy during evaluation/inference.

## 5. What the policy can learn

The policy can learn when another expansion is likely to recover answer
entities or supporting chunks based on retrieval-state features, while
accounting for the expansion cost. It does not learn to choose a graph relation
or a frontier node directly; the environment always expands the most
query-similar frontier entity.

Gold answer and supporting-chunk labels are used by the environment to compute
training rewards and evaluation metrics. They are not included in the 10
inference-state features. However, the current training QA records may contain
gold seed entities unless `--link` is enabled. Use `--link` to start from
question entity linking instead of those QA-provided seed entities.

## 6. Parameter search objective

The `train.py --search-params` option tests:

- `max_steps`: 2, 4, 6, 8
- `step_cost`: 0.01, 0.05, 0.10

For each pair it trains a candidate policy and evaluates it on the dev set (or
the train set if the dev set is empty). The selected candidate maximizes:

\[
\text{search score}
= \text{support-recall@K}
+ \text{answer-in-graph}
- 0.02\times\text{average expansion steps}
\]

The selected settings are then used for a fresh training run and written to
`rl_params.json` beside the policy file. This is a project-specific tuning
objective, not the REINFORCE training objective.

## 7. Relationship to the detailed metrics report

`rag_metrics.py` has a separate reporting efficiency score:

\[
\text{report score}
= \text{support-recall@K}
+ \text{answer-in-top-chunks}
- 0.02\times\text{average expansion steps}
\]

This is not used to update the policy weights. It is a summary score for
comparing dense, never-expand, always-expand, and learned-RL retrieval. The
report's `answer_in_top_chunks` uses normalized string containment, not an
answer-generation model or semantic answer evaluation.

## 8. Run commands

From the repository root, train and tune a policy:

```bash
python kg_interface/train.py \
  --data kg_interface/data \
  --epochs 5 \
  --k 3 \
  --link \
  --search-params \
  --train-limit 2000 \
  --dev-limit 500 \
  --out kg_interface/policy.npz
```

Evaluate the saved policy:

```bash
python kg_interface/rag_metrics.py \
  --data kg_interface/data \
  --policy kg_interface/policy.npz \
  --params kg_interface/rl_params.json \
  --split dev \
  --k 3 \
  --output kg_interface/metrics
```
