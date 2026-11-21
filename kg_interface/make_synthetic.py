"""Make a bigger fake KG (same 5-file format) so the RL code can be tested before the real graph exists.
   python make_synthetic.py synth_data 300"""
import json, os, random, sys
out, n = (sys.argv[1] if len(sys.argv) > 1 else "synth_data"), int(sys.argv[2]) if len(sys.argv) > 2 else 300
random.seed(0)
syl = ["ka", "ro", "mi", "ta", "ve", "lu", "sa", "ni", "do", "pe", "zu", "ha", "bo", "re", "gi"]
used = set()
def word():
    while True:
        w = "".join(random.choices(syl, k=3)).capitalize()
        if w not in used:
            used.add(w); return w
os.makedirs(out, exist_ok=True)
E, T, C, Q = [], [], [], []
def ent(name, desc):
    E.append((f"E{len(E)+1:04d}", name, desc)); return E[-1][0]
def chunk(doc, text, ents):
    C.append({"chunk_id": f"C{len(C)+1:04d}", "document_id": doc, "text": text, "entity_ids": ents}); return C[-1]["chunk_id"]
for i in range(n):
    co, ci, la, lm = word(), word(), word(), word()
    ce = ent(co, f"{co} is a country. Its capital is {ci} and its official language is {la}.")
    cie = ent(ci, f"{ci} is a city, the capital of {co}.")
    le = ent(la, f"{la} is a language, official in {co}.")
    me = ent(lm, f"{lm} is a famous landmark located in the city of {ci}.")
    T += [(cie, "R001", ce), (le, "R002", ce), (me, "R003", cie)]
    c1 = chunk(f"D{i}", f"{ci} is the capital of {co}.", [cie, ce])
    c2 = chunk(f"D{i}", f"{la} is the official language of {co}.", [le, ce])
    c3 = chunk(f"D{i}", f"The {lm} is a famous landmark in {ci}.", [me, cie])
    Q += [
        (f"What is the capital of {co}?", ci, [ce], [cie], [c1]),
        (f"Which language is official in the country whose capital is {ci}?", la, [cie], [le], [c1, c2]),
        (f"Which language is official in the country containing the landmark {lm}?", la, [me], [le], [c3, c1, c2]),
    ]
random.shuffle(Q)
w = lambda name, rows, hdr: open(os.path.join(out, name), "w", encoding="utf-8").write("\t".join(hdr) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
w("entities.tsv", E, ["entity_id", "name", "description"])
w("relations.tsv", [("R001", "capital_of"), ("R002", "official_language_of"), ("R003", "landmark_in")], ["relation_id", "name"])
w("triples.tsv", T, ["head_entity_id", "relation_id", "tail_entity_id"])
open(os.path.join(out, "chunks.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(c) for c in C) + "\n")
with open(os.path.join(out, "qa_pairs.jsonl"), "w", encoding="utf-8") as f:
    for i, (q, a, ents, ans, sup) in enumerate(Q):
        split = "train" if i < 0.7 * len(Q) else "dev" if i < 0.85 * len(Q) else "test"
        f.write(json.dumps({"qa_id": f"Q{i+1:05d}", "question": q, "answer": a, "entities": ents, "answer_entity_ids": ans, "supporting_chunk_ids": sup, "split": split}) + "\n")
print(f"wrote {len(E)} entities, {len(T)} triples, {len(C)} chunks, {len(Q)} QA to {out}")
