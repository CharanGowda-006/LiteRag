"""Convert HotpotQA parquet shards to the LiteRag KG interface format."""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


SEPARATOR = re.compile(r"[^\w]+")
PAREN_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")
CAPITALIZED = re.compile(
    r"\b[A-Z][\w'.\-]*(?:\s+(?:(?:of|the|de|la|le|van|von)\s+)?"
    r"[A-Z][\w'.\-]*)*"
)


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = SEPARATOR.sub(" ", text.lower()).strip()
    return text[4:] if text.startswith("the ") else text


def load_split(data_dir, pattern):
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No HotpotQA shards matching {Path(data_dir) / pattern}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data")
    parser.add_argument("--out", default="data")
    parser.add_argument("--sample-train", type=int, default=20_000)
    parser.add_argument("--sample-validation", type=int, default=7_405)
    parser.add_argument("--ner-model", default="en_core_web_sm")
    args = parser.parse_args()

    train = load_split(args.data, "train-*.parquet")
    validation = load_split(args.data, "validation-*.parquet")
    if args.sample_train:
        train = train.head(min(args.sample_train, len(train)))
    if args.sample_validation:
        validation = validation.head(min(args.sample_validation, len(validation)))

    train = train.copy()
    validation = validation.copy()
    train["split"] = "train"
    validation["split"] = "validation"
    frame = pd.concat([train, validation], ignore_index=True)
    frame = frame.drop_duplicates("id", keep="first").reset_index(drop=True)

    documents = {}
    document_titles = []
    document_sentences = []
    question_documents = defaultdict(set)

    for question_index, context in enumerate(frame["context"]):
        for title, sentences in zip(context["title"], context["sentences"]):
            title = str(title).strip()
            if title not in documents:
                documents[title] = len(document_titles)
                document_titles.append(title)
                document_sentences.append([str(sentence).strip() for sentence in sentences])
            question_documents[question_index].add(documents[title])

    chunks = []
    chunk_lookup = {}
    chunk_entities = []
    for document_index, title in enumerate(document_titles):
        for sentence_index, text in enumerate(document_sentences[document_index]):
            chunk_id = f"C{len(chunks) + 1:06d}"
            chunk_lookup[(document_index, sentence_index)] = len(chunks)
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": f"D{document_index + 1:06d}",
                "text": text,
                "entity_ids": [],
            })
            chunk_entities.append(set([title]))

    entity_names = {}
    entity_descriptions = {}
    for title in document_titles:
        for name in {title, PAREN_SUFFIX.sub("", title).strip()}:
            if name and normalize(name) not in entity_names:
                entity_names[normalize(name)] = name
                entity_descriptions[normalize(name)] = f"HotpotQA article: {name}"

    texts = [chunk["text"] for chunk in chunks]
    try:
        import spacy
        nlp = spacy.load(args.ner_model, disable=["parser", "lemmatizer", "textcat"])
        extracted = []
        kept_labels = {"PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
                       "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
        for doc in nlp.pipe(texts, batch_size=256):
            extracted.append([ent.text for ent in doc.ents if ent.label_ in kept_labels])
    except (ImportError, OSError):
        print("spaCy model unavailable; using article-title entities only.")
        extracted = [[] for _ in texts]

    for chunk_index, names in enumerate(extracted):
        for name in names:
            key = normalize(name)
            if key:
                entity_names.setdefault(key, name)
                entity_descriptions.setdefault(key, f"Named entity: {name}")
                chunk_entities[chunk_index].add(name)

    title_lookup = {}
    for title in document_titles:
        title_lookup.setdefault(normalize(title), title)
        core = PAREN_SUFFIX.sub("", title).strip()
        title_lookup.setdefault(normalize(core), title)

    for chunk_index, names in enumerate(chunk_entities):
        chunk = chunks[chunk_index]
        ids = set()
        for name in names:
            key = normalize(name)
            if key in entity_names:
                canonical = title_lookup.get(key, entity_names[key])
                canonical_key = normalize(canonical)
                if canonical_key in entity_names:
                    ids.add(canonical_key)
        for name in extracted[chunk_index]:
            key = normalize(name)
            if key in entity_names:
                ids.add(key)
        chunk["entity_ids"] = sorted(ids)

    entities = []
    entity_id_by_key = {}
    for key, name in entity_names.items():
        entity_id = f"E{len(entities) + 1:07d}"
        entity_id_by_key[key] = entity_id
        entities.append({
            "entity_id": entity_id,
            "name": name,
            "description": entity_descriptions.get(key, f"HotpotQA entity: {name}"),
        })

    entity_chunks = defaultdict(set)
    for chunk in chunks:
        entity_ids = []
        for key in chunk["entity_ids"]:
            entity_id = entity_id_by_key.get(key)
            if entity_id:
                entity_ids.append(entity_id)
                entity_chunks[entity_id].add(chunk["chunk_id"])
        chunk["entity_ids"] = sorted(set(entity_ids))

    relations = [
        ("R001", "co_supporting_document"),
        ("R002", "answer_document"),
        ("R003", "linked_from_sentence"),
    ]
    triples = set()
    relations_by_name = {name: relation_id for relation_id, name in relations}

    for document_index, title in enumerate(document_titles):
        key = normalize(title)
        entity_id = entity_id_by_key.get(key)
        if not entity_id:
            continue
        core = normalize(PAREN_SUFFIX.sub("", title).strip())
        for chunk in document_sentences[document_index]:
            pass

    for question_index, documents_for_question in enumerate(question_documents.values()):
        ids = sorted(documents_for_question)
        for left_index in range(len(ids)):
            for right_index in range(left_index + 1, len(ids)):
                left_key = normalize(document_titles[ids[left_index]])
                right_key = normalize(document_titles[ids[right_index]])
                left_id = entity_id_by_key.get(left_key)
                right_id = entity_id_by_key.get(right_key)
                if left_id and right_id:
                    triples.add((left_id, relations_by_name["co_supporting_document"], right_id))

    qa_pairs = []
    for question_index, row in frame.iterrows():
        supporting_chunks = []
        context_titles = set(row["context"]["title"])
        question_entity_ids = []
        for title in context_titles:
            key = normalize(title)
            entity_id = entity_id_by_key.get(key)
            if entity_id:
                question_entity_ids.append(entity_id)

        for title, sentence_index in zip(row["supporting_facts"]["title"],
                                         row["supporting_facts"]["sent_id"]):
            document_index = documents.get(str(title).strip())
            if document_index is not None:
                chunk_index = chunk_lookup.get((document_index, int(sentence_index)))
                if chunk_index is not None:
                    supporting_chunks.append(chunks[chunk_index]["chunk_id"])

        answer = normalize(row["answer"])
        answer_entity_ids = []
        if answer:
            answer_key = answer
            if answer_key in entity_id_by_key:
                answer_entity_ids.append(entity_id_by_key[answer_key])

        qa_split = "train" if row["split"] == "train" else "dev"
        qa_pairs.append({
            "qa_id": f"Q{question_index + 1:07d}",
            "question": str(row["question"]).strip(),
            "answer": str(row["answer"]).strip(),
            "entities": sorted(set(question_entity_ids)),
            "answer_entity_ids": sorted(set(answer_entity_ids)),
            "supporting_chunk_ids": sorted(set(supporting_chunks)),
            "split": qa_split,
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write_tsv(name, rows, columns):
        frame_out = pd.DataFrame(rows, columns=columns)
        frame_out.to_csv(out / name, sep="\t", index=False)

    write_tsv("entities.tsv",
              [(row["entity_id"], row["name"], row["description"]) for row in entities],
              ["entity_id", "name", "description"])
    write_tsv("relations.tsv", relations, ["relation_id", "name"])
    write_tsv("triples.tsv",
              [(head, relation, tail) for head, relation, tail in triples],
              ["head_entity_id", "relation_id", "tail_entity_id"])

    with (out / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    with (out / "qa_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for qa in qa_pairs:
            handle.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entities):,} entities, {len(triples):,} triples, "
          f"{len(chunks):,} chunks, {len(qa_pairs):,} QA pairs to {out}")


if __name__ == "__main__":
    main()
