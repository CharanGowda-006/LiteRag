"""Validate KG interface files. Usage: python validate_kg.py data/"""
import csv, json, sys, os
import numpy as np  # only used if embeddings exist

d = sys.argv[1] if len(sys.argv) > 1 else "data"
errs = []

def tsv(name):
    with open(os.path.join(d, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def jsonl(name):
    with open(os.path.join(d, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

ents, rels, trips = tsv("entities.tsv"), tsv("relations.tsv"), tsv("triples.tsv")
chunks, qa = jsonl("chunks.jsonl"), jsonl("qa_pairs.jsonl")

def unique(rows, key, label):
    ids = [r[key] for r in rows]
    if len(ids) != len(set(ids)): errs.append(f"duplicate {label} ids")
    return set(ids)

E, R = unique(ents, "entity_id", "entity"), unique(rels, "relation_id", "relation")
C = unique(chunks, "chunk_id", "chunk"); unique(qa, "qa_id", "qa")
for r in ents + rels:
    if any(not v for v in r.values()): errs.append(f"empty field: {r}")
seen = set()
for t in trips:
    h, r, tl = t["head_entity_id"], t["relation_id"], t["tail_entity_id"]
    if h not in E or tl not in E: errs.append(f"unknown entity in triple {t}")
    if r not in R: errs.append(f"unknown relation in triple {t}")
    if (h, r, tl) in seen: errs.append(f"duplicate triple {t}")
    seen.add((h, r, tl))
for c in chunks:
    for e in c["entity_ids"]:
        if e not in E: errs.append(f"{c['chunk_id']}: unknown entity {e}")
for q in qa:
    for e in q["entities"] + q.get("answer_entity_ids", []):
        if e not in E: errs.append(f"{q['qa_id']}: unknown entity {e}")
    for c in q.get("supporting_chunk_ids", []):
        if c not in C: errs.append(f"{q['qa_id']}: unknown chunk {c}")
# optional embeddings: row i == i-th row of entities.tsv / chunks.jsonl
for fname, n in [("entity_embeddings.npy", len(ents)), ("chunk_embeddings.npy", len(chunks))]:
    p = os.path.join(d, fname)
    if os.path.exists(p) and np.load(p, mmap_mode="r").shape[0] != n:
        errs.append(f"{fname} row count != {n}")

print("OK" if not errs else "\n".join(errs))
print(f"{len(ents)} entities, {len(rels)} relations, {len(trips)} triples, {len(chunks)} chunks, {len(qa)} QA")
sys.exit(1 if errs else 0)
