import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from score_preference_embedding_cosines import (
    DEFAULT_EMBEDDING_MODEL,
    OpenAIEmbeddingClient,
    format_completion,
    format_system_prompt,
    l2_normalize,
    load_dotenv,
    load_text_cache,
    response_pair_length,
    save_text_cache,
    stable_text_id,
)


DEFAULT_LOGPROBS_PATH = Path(
    "experiments/original-dataset/inverse/"
    "ultrafeedback_mean_aspects_first10k_first15_per_category_logprobs.jsonl"
)
DEFAULT_RESPONSE_EMBEDDINGS_PATH = Path(
    "experiments/original-dataset/inverse/original_logprob_embeddings.npz"
)
DEFAULT_RESPONSE_CACHE_PATH = Path(
    "experiments/original-dataset/inverse/"
    "ultrafeedback_mean_aspects_first10k_first15_per_category_embedding_cache."
    "text-embedding-3-large.npz"
)
DEFAULT_SYSTEM_CACHE_PATH = Path(
    "experiments/dog-lls-q0.1-trunc20/embedding_cosines/"
    "embedding_cache_by_text.text-embedding-3-large.npz"
)
DEFAULT_OUTPUT_PATH = Path(
    "experiments/original-dataset/inverse/"
    "ultrafeedback_mean_aspects_first10k_first15_per_category_logprob_embeddings.npz"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a row-aligned compact embedding file for the SVD MLP fit by "
            "joining cached response embeddings, cached system embeddings, and "
            "length denominators from the normalized original logprob cache."
        )
    )
    parser.add_argument("--logprobs-path", type=Path, default=DEFAULT_LOGPROBS_PATH)
    parser.add_argument(
        "--response-embeddings-path",
        type=Path,
        default=DEFAULT_RESPONSE_EMBEDDINGS_PATH,
        help="Compact embedding file containing User/Assistant response embeddings.",
    )
    parser.add_argument(
        "--response-cache-path",
        type=Path,
        default=DEFAULT_RESPONSE_CACHE_PATH,
        help="Text-id keyed response embedding cache. Missing texts are added here.",
    )
    parser.add_argument(
        "--system-cache-path",
        type=Path,
        default=DEFAULT_SYSTEM_CACHE_PATH,
        help="Text-id keyed cache containing system prompt embeddings.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="L2-normalize embeddings before saving. Defaults to preserving cached vectors.",
    )
    return parser.parse_args()


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def load_response_embedding_lookup(path):
    payload = np.load(path, allow_pickle=False)
    texts = np.asarray(payload["unique_texts"]).astype(str)
    embeddings = np.asarray(payload["unique_embeddings"], dtype=np.float32)
    return {text: embeddings[index] for index, text in enumerate(texts)}, embeddings.shape[1]


def load_lenient_text_cache(path, model):
    try:
        return load_text_cache(path, model)
    except KeyError:
        payload = np.load(path, allow_pickle=False)
        text_ids = [str(text_id) for text_id in payload["text_ids"]]
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        return dict(zip(text_ids, embeddings, strict=True))


def load_system_embedding_lookup(path):
    payload = np.load(path, allow_pickle=False)
    text_ids = np.asarray(payload["text_ids"]).astype(str)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    return {text_id: embeddings[index] for index, text_id in enumerate(text_ids)}, embeddings.shape[1]


def add_unique(text, embedding, index_by_text, unique_texts, unique_embeddings):
    index = index_by_text.get(text)
    if index is None:
        index = len(unique_texts)
        index_by_text[text] = index
        unique_texts.append(text)
        unique_embeddings.append(embedding)
    return index


def collect_required_texts(path):
    system_texts = set()
    response_texts = set()
    num_rows = 0
    for _, row in read_jsonl(path):
        num_rows += 1
        system_texts.add(format_system_prompt(row["s"]))
        response_texts.add(format_completion(row["p"], row["r_plus"]))
        response_texts.add(format_completion(row["p"], row["r_minus"]))
    return system_texts, response_texts, num_rows


def embed_missing_texts(texts, cache, args):
    missing = [text for text in sorted(texts) if stable_text_id(text) not in cache]
    if not missing:
        return 0

    load_dotenv(args.env_file)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env or set it in the environment.")

    client = OpenAIEmbeddingClient(
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    print(f"Embedding {len(missing)} missing texts with {args.model}")
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start:start + args.batch_size]
        vectors = client.embed(batch)
        for text, vector in zip(batch, vectors, strict=True):
            cache[stable_text_id(text)] = np.asarray(vector, dtype=np.float32)
        print(f"  embedded {min(start + len(batch), len(missing))}/{len(missing)}")
    save_text_cache(args.response_cache_path, cache, args.model)
    return len(missing)


def main():
    args = parse_args()
    response_lookup, response_dim = load_response_embedding_lookup(args.response_embeddings_path)
    response_cache = load_lenient_text_cache(args.response_cache_path, args.model)
    system_lookup, system_dim = load_system_embedding_lookup(args.system_cache_path)
    if response_dim != system_dim:
        raise ValueError(f"Embedding dimensions differ: responses={response_dim}, systems={system_dim}")

    system_texts, response_texts, expected_rows = collect_required_texts(args.logprobs_path)
    missing_systems = [
        text for text in sorted(system_texts)
        if stable_text_id(text) not in system_lookup
    ]
    if missing_systems:
        raise ValueError(
            f"System cache is missing {len(missing_systems)} system prompts. "
            f"First missing: {missing_systems[0]!r}"
        )
    embedded_missing = embed_missing_texts(response_texts, response_cache, args)

    index_by_text = {}
    unique_texts = []
    unique_embeddings = []
    system_indices = []
    chosen_indices = []
    rejected_indices = []
    targets = []
    raw_targets = []
    length_denominators = []
    missing_responses = []

    for line_number, row in read_jsonl(args.logprobs_path):
        system_text = format_system_prompt(row["s"])
        chosen_text = format_completion(row["p"], row["r_plus"])
        rejected_text = format_completion(row["p"], row["r_minus"])

        system_embedding = system_lookup.get(stable_text_id(system_text))
        chosen_embedding = response_lookup.get(chosen_text)
        if chosen_embedding is None:
            chosen_embedding = response_cache.get(stable_text_id(chosen_text))
        rejected_embedding = response_lookup.get(rejected_text)
        if rejected_embedding is None:
            rejected_embedding = response_cache.get(stable_text_id(rejected_text))
        length_denominator = response_pair_length(row["r_plus"], row["r_minus"])

        if chosen_embedding is None:
            missing_responses.append((line_number, "r_plus"))
        if rejected_embedding is None:
            missing_responses.append((line_number, "r_minus"))
        if chosen_embedding is None or rejected_embedding is None:
            continue

        raw_margin = float(row["logprob"])
        system_indices.append(
            add_unique(system_text, system_embedding, index_by_text, unique_texts, unique_embeddings)
        )
        chosen_indices.append(
            add_unique(chosen_text, chosen_embedding, index_by_text, unique_texts, unique_embeddings)
        )
        rejected_indices.append(
            add_unique(rejected_text, rejected_embedding, index_by_text, unique_texts, unique_embeddings)
        )
        targets.append(raw_margin / max(length_denominator, 1))
        raw_targets.append(raw_margin)
        length_denominators.append(length_denominator)

    if missing_responses:
        examples = {
            "missing_responses": missing_responses[:5],
        }
        raise ValueError(
            "Could not build a complete row-aligned embedding file. "
            f"Missing responses={len(missing_responses)}. "
            f"Examples: {examples}"
        )
    if len(targets) != expected_rows:
        raise ValueError(f"Built {len(targets)} rows, expected {expected_rows}.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_embeddings_array = np.asarray(unique_embeddings, dtype=np.float32)
    if args.normalize_embeddings:
        unique_embeddings_array = l2_normalize(unique_embeddings_array).astype(np.float32)
    np.savez_compressed(
        args.output_path,
        target_logprob_margins=np.asarray(targets, dtype=np.float32),
        raw_logprob_margins=np.asarray(raw_targets, dtype=np.float32),
        length_denominators=np.asarray(length_denominators, dtype=np.int32),
        system_text_indices=np.asarray(system_indices, dtype=np.int64),
        chosen_text_indices=np.asarray(chosen_indices, dtype=np.int64),
        rejected_text_indices=np.asarray(rejected_indices, dtype=np.int64),
        unique_embeddings=unique_embeddings_array,
        unique_texts=np.asarray(unique_texts),
    )

    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "logprobs_path": str(args.logprobs_path),
        "response_embeddings_path": str(args.response_embeddings_path),
        "response_cache_path": str(args.response_cache_path),
        "system_cache_path": str(args.system_cache_path),
        "output_path": str(args.output_path),
        "num_rows": len(targets),
        "expected_rows": expected_rows,
        "num_unique_texts": len(unique_texts),
        "embedding_dim": int(unique_embeddings_array.shape[1]),
        "num_required_system_texts": len(system_texts),
        "num_required_response_texts": len(response_texts),
        "num_texts_embedded_this_run": embedded_missing,
        "length_denominator": "len(r_plus.split()) + len(r_minus.split())",
        "target_logprob_margins": "raw_logprob_margin / whitespace_length_denominator",
        "normalize_embeddings": args.normalize_embeddings,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.output_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
