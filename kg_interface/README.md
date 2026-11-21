# KG interface spec (LiteRag)

All files in `data/`, UTF-8, TSV with header row, no tabs/newlines inside fields.

| File | Columns / keys |
|---|---|
| triples.tsv | head_entity_id, relation_id, tail_entity_id |
| entities.tsv | entity_id, name, description |
| relations.tsv | relation_id, name |
| chunks.jsonl | chunk_id, document_id, text, entity_ids |
| qa_pairs.jsonl | qa_id, question, answer, entities, answer_entity_ids, supporting_chunk_ids, split |

ID conventions: entities `E0001`, relations `R001`, chunks `C0001`, documents `D001`, QA `Q0001`.
Every ID used anywhere must exist in its master file (entities.tsv / relations.tsv / chunks.jsonl).

Optional embeddings (float32): `entity_embeddings.npy` (row i = i-th row of entities.tsv),
`chunk_embeddings.npy` (row i = i-th line of chunks.jsonl). Record the model name in `embeddings_meta.json`.

RL conventions: triples are directed, but neighbour expansion treats the graph as undirected
(add inverse edges at load time). Actions: EXPAND(node) / STOP. Reward uses
answer_entity_ids and supporting_chunk_ids from qa_pairs.jsonl.

Check with: `python validate_kg.py data/`
