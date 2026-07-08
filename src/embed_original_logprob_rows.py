import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from score_preference_embedding_cosines import (
    DEFAULT_EMBEDDING_MODEL,
    OpenAIEmbeddingClient,
    load_dotenv,
    load_text_cache,
    save_text_cache,
    stable_text_id,
)


DEFAULT_INPUT_PATH = Path("experiments/original-dataset/inverse/original_logprobs.jsonl")
DEFAULT_OUTPUT_NAME = "original_logprob_embeddings.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Call the OpenAI embeddings API for row-aligned e(s), e(p,r+), "
            "and e(p,r-) from original_logprobs.jsonl."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="JSONL file with s, p, r_plus, r_minus rows.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output .npz path. Defaults to original_logprob_embeddings.npz "
            "next to --input-path."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Output metadata JSON path. Defaults next to the .npz file.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help=(
            "Incremental per-text embedding cache. Defaults to "
            "original_logprob_embedding_cache.<model>.npz next to --input-path."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="OpenAI embedding model.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Storage dtype for saved embedding arrays.",
    )
    parser.add_argument(
        "--save-row-arrays",
        action="store_true",
        help=(
            "Also save expanded row-aligned embedding arrays. By default the "
            "script saves compact unique_embeddings plus row indices."
        ),
    )
    return parser.parse_args()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc


def system_embedding_text(system_prompt):
    return f"System: {system_prompt}"


def response_embedding_text(prompt, response):
    return f"User: {prompt}\nAssistant: {response}"


def paired_rows(path):
    rows = []
    skipped = 0
    required = {"s", "p", "r_plus", "r_minus"}
    for line_number, row in read_jsonl(path):
        if not required.issubset(row):
            skipped += 1
            continue
        rows.append(
            {
                "line_number": line_number,
                "s": row["s"],
                "p": row["p"],
                "r_plus": row["r_plus"],
                "r_minus": row["r_minus"],
                "target_logprob_margin": float(row["logprob"])
                if "logprob" in row
                else np.nan,
                "raw_logprob_margin": float(row["raw_logprob_margin"])
                if "raw_logprob_margin" in row
                else np.nan,
                "length_denominator": int(row["length_denominator"])
                if "length_denominator" in row
                else 0,
                "score_normalization": row.get("score_normalization"),
                "system_prompt_index": row.get("system_prompt_index"),
                "category": row.get("category"),
                "trait": row.get("trait"),
                "trait_normalized": row.get("trait_normalized"),
                "key": row.get("key"),
            }
        )
    if not rows:
        raise ValueError(f"No paired s/p/r_plus/r_minus rows found in {path}")
    return rows, skipped


def unique_text_indices(texts):
    index_by_text = {}
    unique_texts = []
    inverse = []
    for text in texts:
        index = index_by_text.get(text)
        if index is None:
            index = len(unique_texts)
            index_by_text[text] = index
            unique_texts.append(text)
        inverse.append(index)
    return unique_texts, np.asarray(inverse, dtype=np.int64)


def model_cache_name(model):
    return model.replace("/", "_").replace("\\", "_").replace(":", "_")


def embed_with_cache(client, texts, text_ids, batch_size, cache_path, model):
    cache = load_text_cache(cache_path, model) if cache_path.exists() else {}
    if cache:
        print(f"Loaded embedding cache: {cache_path} ({len(cache)} texts)")

    missing = [
        (text, text_id)
        for text, text_id in zip(texts, text_ids, strict=True)
        if text_id not in cache
    ]
    print(f"Need {len(missing)} new embeddings; reusing {len(texts) - len(missing)} cached embeddings.")

    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        batch_texts = [text for text, _ in batch]
        batch_ids = [text_id for _, text_id in batch]
        stop = start + len(batch)
        print(f"Embedding missing texts {start + 1}-{stop} of {len(missing)}", flush=True)
        embeddings = client.embed(batch_texts)
        for text_id, embedding in zip(batch_ids, embeddings, strict=True):
            cache[text_id] = embedding
        save_text_cache(cache_path, cache, model)
        print(f"Saved embedding cache checkpoint: {cache_path}", flush=True)

    return np.asarray([cache[text_id] for text_id in text_ids], dtype=np.float64)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    args = parse_args()
    output_path = args.output_path or args.input_path.with_name(DEFAULT_OUTPUT_NAME)
    metadata_path = args.metadata_path or output_path.with_suffix(".summary.json")
    cache_path = args.cache_path or args.input_path.with_name(
        f"original_logprob_embedding_cache.{model_cache_name(args.model)}.npz"
    )

    rows, skipped = paired_rows(args.input_path)
    print(f"Loaded {len(rows)} paired rows from {args.input_path}")
    if skipped:
        print(f"Skipped {skipped} non-paired rows")

    system_texts = [system_embedding_text(row["s"]) for row in rows]
    chosen_texts = [
        response_embedding_text(row["p"], row["r_plus"])
        for row in rows
    ]
    rejected_texts = [
        response_embedding_text(row["p"], row["r_minus"])
        for row in rows
    ]
    all_texts = system_texts + chosen_texts + rejected_texts
    unique_texts, inverse_indices = unique_text_indices(all_texts)
    unique_text_ids = [stable_text_id(text) for text in unique_texts]
    num_rows = len(rows)
    system_indices = inverse_indices[:num_rows]
    chosen_indices = inverse_indices[num_rows:2 * num_rows]
    rejected_indices = inverse_indices[2 * num_rows:]

    print(
        f"Prepared {len(all_texts)} row texts "
        f"({len(unique_texts)} unique) for {len(rows)} rows"
    )

    load_dotenv(args.env_file)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env or set it in the environment.")

    print(f"Embedding unique texts with OpenAI model {args.model}")
    client = OpenAIEmbeddingClient(
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    unique_embeddings = embed_with_cache(
        client,
        unique_texts,
        unique_text_ids,
        args.batch_size,
        cache_path,
        args.model,
    )

    storage_dtype = np.float16 if args.dtype == "float16" else np.float32
    targets = np.asarray(
        [row["target_logprob_margin"] for row in rows],
        dtype=np.float32,
    )
    raw_targets = np.asarray(
        [row["raw_logprob_margin"] for row in rows],
        dtype=np.float32,
    )
    length_denominators = np.asarray(
        [row["length_denominator"] for row in rows],
        dtype=np.int32,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_arrays = {
        "target_logprob_margins": targets,
        "raw_logprob_margins": raw_targets,
        "length_denominators": length_denominators,
        "system_text_indices": system_indices,
        "chosen_text_indices": chosen_indices,
        "rejected_text_indices": rejected_indices,
        "unique_embeddings": unique_embeddings.astype(storage_dtype),
        "unique_texts": np.asarray(unique_texts),
    }
    if args.save_row_arrays:
        output_arrays.update(
            {
                "system_embeddings": unique_embeddings[system_indices].astype(storage_dtype),
                "chosen_embeddings": unique_embeddings[chosen_indices].astype(storage_dtype),
                "rejected_embeddings": unique_embeddings[rejected_indices].astype(storage_dtype),
                "response_diff_embeddings": (
                    unique_embeddings[chosen_indices] - unique_embeddings[rejected_indices]
                ).astype(storage_dtype),
            }
        )
    np.savez(output_path, **output_arrays)

    metadata = {
        "created_at": now_iso(),
        "input_path": str(args.input_path),
        "output_path": str(output_path),
        "cache_path": str(cache_path),
        "embedding_provider": "openai",
        "embedding_model": args.model,
        "embedding_batch_size": args.batch_size,
        "storage_dtype": args.dtype,
        "num_rows": len(rows),
        "num_row_texts": len(all_texts),
        "num_unique_texts": len(unique_texts),
        "skipped_rows": skipped,
        "format": "compact_unique_embeddings",
        "save_row_arrays": args.save_row_arrays,
        "arrays": {
            "unique_embeddings": list(unique_embeddings.shape),
            "system_text_indices": list(system_indices.shape),
            "chosen_text_indices": list(chosen_indices.shape),
            "rejected_text_indices": list(rejected_indices.shape),
            "target_logprob_margins": list(targets.shape),
            "raw_logprob_margins": list(raw_targets.shape),
            "length_denominators": list(length_denominators.shape),
        },
        "text_format": {
            "system": "System: {s}",
            "response": "User: {p}\\nAssistant: {r}",
        },
    }
    write_json(metadata_path, metadata)

    print(f"Saved embeddings to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
