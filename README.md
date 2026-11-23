# LiteRag: HotpotQA graph retrieval and RL

This folder contains a small, runnable graph-retrieval system for HotpotQA. It
converts the repository's parquet dataset into a graph/chunk/QA interface, trains
a policy that decides whether to expand the graph or stop, and reports retrieval
metrics against simple baselines.

The current system evaluates **retrieval**, not generated answers: there is no
language-model answer-generation or answer-faithfulness stage yet.

## Project layout

| Path | Purpose |
|---|---|
| `data/` | HotpotQA train and validation parquet shards used by the converter |
| `notebooks/01_data_exploration.ipynb` | Explore the source data and export it for LiteRag |
| `notebooks/02_knowledge_graph.ipynb` | Load and inspect the LiteRag graph |
| `notebooks/03_baseline_retrieval.ipynb` | Run dense-retrieval baseline |
| `notebooks/04_rl_agent.ipynb` | Train the RL policy and search expansion parameters |
| `notebooks/05_evaluation.ipynb` | Compare baselines and the learned policy |
| `notebooks/06_full_hotpotqa_colab.ipynb` | Convert the full dataset and evaluate the full validation split in Colab |
| `build_hotpot_kg.py` | Convert HotpotQA parquet shards to the interface files |
| `data/` | Generated graph, chunks, and QA data consumed by LiteRag |
| `litrag/kg.py` | Load graph data, embeddings, and link question entities |
| `litrag/env.py` | Retrieval environment, state features, rewards, and actions |
| `litrag/agent.py` | Logistic policy and REINFORCE training |
| `litrag/retrieve.py` | Run policy rollouts and retrieve top chunks |
| `litrag/evaluate.py` | Basic dense and graph-policy evaluation helpers |
| `train.py` | Command-line policy training, parameter search, and comparison |
| `rag_metrics.py` | Detailed metrics report in JSON and CSV |
| `validate_kg.py` | Check generated files, references, IDs, and optional embeddings |
| `make_synthetic.py` | Generate a small artificial KG for a quick code-path test |

## Data flow

1. Notebook 01 loads all train and validation parquet shards and creates the
   normalized node/edge tables used for graph exploration.
2. Its final export cell calls `build_hotpot_kg.py` to generate the LiteRag
   interface files in `kg_interface/data/`.
3. Notebooks 02 and 03 load that interface and inspect the graph or evaluate
   dense retrieval.
4. Notebook 04 (or `train.py`) trains a policy using train QA and tunes the
   expansion budget and step cost using dev QA.
5. Notebook 05 (or `rag_metrics.py`) evaluates retrieval on a selected split.

The converter maps HotpotQA validation rows to the interface split name `dev`.
It does not create a separate test split. Its default sample sizes are 20,000
train rows and all 7,405 validation rows. Pass a sample size of `0` to use the
entire corresponding split.

## Full HotpotQA run in Google Colab

Use [06_full_hotpotqa_colab.ipynb](notebooks/06_full_hotpotqa_colab.ipynb) to
convert all train and validation shards, train a policy, and evaluate all
validation questions. It expects the three parquet shards in
`MyDrive/HotpotQA/data/` and writes the generated data, policy, and metrics to
`MyDrive/LiteRag_Full_Run/`.

The current hash-embedding implementation holds dense 2,048-dimensional
matrices in memory. The full corpus needs about 14.4 GiB for chunk embeddings
and at least 3.9 GiB for title-entity embeddings alone (aliases may add more),
so the notebook requires a Colab High-RAM runtime (at least 40 GiB total; 50 GiB
recommended). The default notebook trains on 1,000 train questions for 2 epochs to limit runtime, but
evaluates on the complete validation split. Set `TRAIN_LIMIT = None` in the
notebook to train on all train questions; this can take many hours.

## Install and prepare data

Run commands below from the repository root unless noted otherwise. The root
`requirements.txt` lists notebook, data, retrieval, and ML dependencies.

The source parquet files are expected in `data/`, with names matching
`train-*.parquet` and `validation-*.parquet`. Convert them with:

```bash
python kg_interface/build_hotpot_kg.py \
  --data data \
  --out kg_interface/data \
  --sample-train 20000 \
  --sample-validation 7405
```

For a quick small conversion:

```bash
python kg_interface/build_hotpot_kg.py \
  --data data \
  --out kg_interface/data \
  --sample-train 500 \
  --sample-validation 100
```

The converter uses spaCy's `en_core_web_sm` model when installed. Without spaCy
or the model it falls back to article-title entities only. The latter is
functional but produces a sparser entity graph. It does not create learned
embeddings; the loader uses its dependency-free `hash-bow` embedder unless
optional saved embeddings and `embeddings_meta.json` are provided.

## LiteRag interface format

All text files use UTF-8. TSV files have headers.

| File | Required fields |
|---|---|
| `entities.tsv` | `entity_id`, `name`, `description` |
| `relations.tsv` | `relation_id`, `name` |
| `triples.tsv` | `head_entity_id`, `relation_id`, `tail_entity_id` |
| `chunks.jsonl` | `chunk_id`, `document_id`, `text`, `entity_ids` |
| `qa_pairs.jsonl` | `qa_id`, `question`, `answer`, `entities`, `answer_entity_ids`, `supporting_chunk_ids`, `split` |

IDs are unique within each type. Every entity, relation, and supporting-chunk
reference must resolve to an ID in its corresponding file. Validation command:

```bash
python kg_interface/validate_kg.py kg_interface/data
```

Optional embeddings are NumPy float32 arrays:

- `entity_embeddings.npy`: row order matches `entities.tsv`.
- `chunk_embeddings.npy`: row order matches `chunks.jsonl`.
- `embeddings_meta.json`: contains the embedding model name in a `model` field.

The graph loader adds inverse adjacency for traversal. Graph triples are stored
as directed `(head, relation, tail)` rows, while retrieval expansion can visit
neighbors in either direction.

## RL retrieval behavior

Each question starts from linked or supplied seed entities. The environment
collects chunks attached to visited entities and exposes 10 state features,
including expansion progress, frontier size, query/entity similarity, best
chunk similarity, and collected chunk count.

At each step the policy chooses:

- `EXPAND`: expand the frontier entity most similar to the question; add its
  unvisited neighbors and their chunks.
- `STOP`: finish retrieval and rank the collected chunks by query similarity.

During training, the reward reflects increases in answer-entity and supporting
chunk coverage, minus the per-expansion step cost. Gold labels are used for
training rewards/evaluation, not as inference-time policy features.

For the state features, action probabilities, reward derivation, REINFORCE
updates, and parameter-search formulas, see [RL_FORMULAS.md](RL_FORMULAS.md).

Train and tune with the command-line interface:

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

`--search-params` tries max-step values 2, 4, 6, and 8 with step costs 0.01,
0.05, and 0.10. It chooses the configuration with the best development score:

```text
support_recall@k + answer_in_graph - 0.02 * avg_steps
```

It then retrains with the selected parameters. The policy is saved to the path
given by `--out`; `rl_params.json` is written beside that policy. Increase or
remove the train/dev limits for a larger run. `--link` selects entity linking
from the question rather than using the QA seed entities.

## Metrics report

Generate a JSON and CSV report:

```bash
python kg_interface/rag_metrics.py \
  --data kg_interface/data \
  --policy kg_interface/policy.npz \
  --params kg_interface/rl_params.json \
  --split dev \
  --k 3 \
  --output kg_interface/metrics
```

Use `--limit N` to evaluate only the first `N` QA rows in the selected split.
If the requested split has no rows, the script currently falls back to all QA
rows. The report compares dense retrieval, never-expand, always-expand, and
learned-RL retrieval (when a policy is supplied).

Metrics in `rag_metrics.json` and `rag_metrics.csv`:

- **Support Recall@K**: retrieved gold supporting chunks divided by all gold
  supporting chunks for a question.
- **Support Precision@K**: retrieved gold supporting chunks divided by K.
- **MRR@K**: reciprocal rank of the first retrieved supporting chunk.
- **NDCG@K**: normalized discounted gain for binary supporting-chunk relevance.
- **Answer in top chunks**: normalized answer text is found in at least one of
  the retrieved chunks; this is a string-match heuristic.
- **Answer in graph**: a gold answer entity was visited during graph retrieval.
- **Average expansion steps**: mean number of graph expansions per question.
- **Average latency**: mean measured retrieval time in seconds.
- **Metrics-report efficiency score**: `support_recall@K + answer_in_top_chunks -
  0.02 * avg_steps`. Dense retrieval has no expansion cost, so its step count is
  treated as zero.

The answer-in-chunks measure is not exact match or F1 on a generated answer.
There is no answer generator, faithfulness metric, or end-to-end LLM evaluation
in this implementation. Latency timings are local measurements and are affected
by hardware, caching, and workload. Dense latency includes encoding and ranking;
graph latency times the rollout but not the final top-chunk ranking.

## Current generated snapshot

The checked-in working data snapshot was generated from 2,000 train and 500
validation examples for development. It contains 2,500 QA rows, 27,399 entities,
110,898 triples, and 97,264 sentence chunks. A saved policy and parameter file
are present. These counts and metric values will change when the converter or
training commands are rerun with different inputs, sample sizes, or settings.

The current metrics report was run on the first 20 dev rows with `K=3`. It
reported learned-RL support recall `0.383`, support precision `0.283`, MRR
`0.525`, NDCG `0.407`, answer-in-top-chunks `0.20`, and average expansion steps
`0.45`. This small evaluation is a smoke-scale result, not a full HotpotQA
benchmark.

## Known limitations and next work

- The current converter uses sentence-level chunks, not larger semantic chunks.
- When spaCy is unavailable, entities are limited to article titles.
- The converter currently creates co-supporting-document triples; the declared
  `answer_document` and `linked_from_sentence` relations are reserved and are
  not populated yet.
- Answer-entity labels are assigned only when a normalized answer matches an
  entity name exactly. Many questions therefore have no answer-entity label.
- No test split is generated; validation is used as `dev`.
- The baseline embedding fallback is hash bag-of-words, not a semantic embedding
  model. Saved embedding generation is not included.
- Metrics are retrieval-focused and use supporting facts as relevance labels.
  Answer generation, exact match/F1, faithfulness, and full-scale benchmark
  evaluation remain future work.
