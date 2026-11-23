"""Load the KG interface files (see README.md) and build graph + embedding structures."""
import csv, json, os, re, zlib
from collections import defaultdict
import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "of", "is", "a", "an", "in", "what", "which", "and", "to", "as", "for", "its", "it", "are", "was"}


def norm_text(s):
    return " ".join(_TOKEN.findall(s.lower()))


class HashEmbedder:
    """Dependency-free bag-of-words embedder (fallback). Replace with a real model later."""
    name = "hash-bow"

    def __init__(self, dim=2048):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall(t.lower()):
                if tok not in _STOP:
                    out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class SentenceEmbedder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
        self.model, self.name = SentenceTransformer(model_name), model_name

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def get_embedder(data_dir):
    """Uses embeddings_meta.json ({"model": "..."}) if present, otherwise the hash embedder."""
    p = os.path.join(data_dir, "embeddings_meta.json")
    if os.path.exists(p):
        name = json.load(open(p))["model"]
        return HashEmbedder() if name == "hash-bow" else SentenceEmbedder(name)
    return HashEmbedder()


def _tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class KnowledgeGraph:
    def __init__(self, data_dir, embedder=None, use_saved_embeddings=True):
        j = lambda n: os.path.join(data_dir, n)
        ents, rels = _tsv(j("entities.tsv")), _tsv(j("relations.tsv"))
        self.chunks, self.qa = _jsonl(j("chunks.jsonl")), _jsonl(j("qa_pairs.jsonl"))

        self.ent_ids = [e["entity_id"] for e in ents]
        self.ent_idx = {e: i for i, e in enumerate(self.ent_ids)}
        self.ent_names = [e["name"] for e in ents]
        self.ent_desc = [e["description"] for e in ents]
        self.rel_names = {r["relation_id"]: r["name"] for r in rels}
        self._norm_names = [" " + norm_text(n) + " " for n in self.ent_names]
        self._norm_name_index = {}
        for i, normalized_name in enumerate(self._norm_names):
            key = normalized_name.strip()
            if not key:
                continue
            previous = self._norm_name_index.get(key)
            if previous is None:
                self._norm_name_index[key] = i
            elif isinstance(previous, list):
                previous.append(i)
            else:
                self._norm_name_index[key] = [previous, i]

        # undirected neighbour lists (inverse edges added so EXPAND can go "backwards")
        nb = [set() for _ in ents]
        with open(j("triples.tsv"), encoding="utf-8", newline="") as handle:
            for t in csv.DictReader(handle, delimiter="\t"):
                h = self.ent_idx[t["head_entity_id"]]
                tl = self.ent_idx[t["tail_entity_id"]]
                r = t["relation_id"]
                nb[h].add(tl)
                nb[tl].add(h)
        self.nbrs = [sorted(s) for s in nb]

        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.chunk_idx = {c: i for i, c in enumerate(self.chunk_ids)}
        self.entity_chunks = defaultdict(list)
        for ci, c in enumerate(self.chunks):
            for e in c["entity_ids"]:
                self.entity_chunks[self.ent_idx[e]].append(ci)

        self.embedder = embedder or get_embedder(data_dir)
        self.ent_emb = self._emb(j("entity_embeddings.npy"), [f"{n}. {d}" for n, d in zip(self.ent_names, self.ent_desc)], use_saved_embeddings)
        self.chunk_emb = self._emb(j("chunk_embeddings.npy"), [c["text"] for c in self.chunks], use_saved_embeddings)

    def _emb(self, path, texts, use_saved):
        if use_saved and os.path.exists(path):
            arr = np.load(path).astype(np.float32)
            if arr.shape[0] == len(texts):
                return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
        return self.embedder.encode(texts)

    def encode_query(self, question):
        return self.embedder.encode([question])[0]

    def link_entities(self, question, k=3):
        """Simple entity linking: names found in the question (longest first), else best embedding match."""
        words = norm_text(question).split()
        hit_ids = []
        for start in range(len(words)):
            for end in range(start + 1, len(words) + 1):
                match = self._norm_name_index.get(" ".join(words[start:end]))
                if isinstance(match, list):
                    hit_ids.extend(match)
                elif match is not None:
                    hit_ids.append(match)
        hits = sorted(
            ((len(self._norm_names[i]), i) for i in set(hit_ids)),
            reverse=True,
        )
        if hits:
            return [i for _, i in hits[:k]]
        return [int(np.argmax(self.ent_emb @ self.encode_query(question)))]
