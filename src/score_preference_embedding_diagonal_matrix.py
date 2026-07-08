import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from hf_sync import pull_hf_artifacts, push_hf_artifacts
from score_preference_embedding_cosines import (
    DEFAULT_DATASET_PATH,
    DEFAULT_DOG_PROMPT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SYSTEM_PROMPTS_PATH,
    OpenAIEmbeddingClient,
    embed_texts,
    format_completion,
    format_system_prompt,
    l2_normalize,
    load_dotenv,
    load_preference_examples,
    load_system_prompts,
    load_text_cache,
    response_pair_length,
    save_text_cache,
    text_ids_for,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = Path("experiments/dog-lls-q0.1-trunc20/embedding_diagonal_matrix")
DEFAULT_COSINE_OUTPUT_DIR = Path("experiments/dog-lls-q0.1-trunc20/embedding_cosines")
DEFAULT_MATRIX_ROOT = Path("experiments/original-dataset/inverse")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score candidate system prompts with a learned diagonal matrix A: "
            "mean_i e(s)^T diag(A) (e(p_i,r_i+) - e(p_i,r_i-))."
        )
    )
    parser.add_argument("--preference-dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--system-prompts-path", type=Path, default=DEFAULT_SYSTEM_PROMPTS_PATH)
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=None,
        help=(
            "Path to A_diagonal.npy. Defaults to the newest "
            "experiments/original-dataset/inverse/diagonal_fit_*/A_diagonal.npy."
        ),
    )
    parser.add_argument(
        "--num-system-prompts",
        type=int,
        default=None,
        help="Number of JSONL system prompts to score before appending --dog-prompt. Defaults to all rows.",
    )
    parser.add_argument("--dog-prompt", default=DEFAULT_DOG_PROMPT)
    parser.add_argument("--model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and overwrite the embedding cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse inputs and print counts without calling the OpenAI API.",
    )
    return parser.parse_args()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def resolve_matrix_path(path):
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Matrix file not found: {path}")
        return path

    candidates = sorted(
        DEFAULT_MATRIX_ROOT.glob("diagonal_fit_*/A_diagonal.npy"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find A_diagonal.npy. Run src/fit_diagonal_logprob_matrix.py "
            "or pass --matrix-path explicitly."
        )
    return candidates[0]


def load_diagonal(path):
    diagonal = np.asarray(np.load(path), dtype=np.float64)
    if diagonal.ndim != 1:
        raise ValueError(f"Expected 1D A_diagonal.npy, got shape {diagonal.shape}.")
    return diagonal


def load_or_embed_texts(args, texts, cache_path):
    text_ids = text_ids_for(texts)
    cache = {} if args.no_cache else load_text_cache(cache_path, args.model)
    if cache:
        print(f"Loaded per-text embedding cache: {cache_path} ({len(cache)} texts)")

    missing_texts = []
    missing_ids = []
    for text, text_id in zip(texts, text_ids, strict=True):
        if text_id not in cache:
            missing_texts.append(text)
            missing_ids.append(text_id)

    if missing_texts:
        load_dotenv(args.env_file)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Missing OPENAI_API_KEY. Add it to .env or set it in the environment.", file=sys.stderr)
            return None

        client = OpenAIEmbeddingClient(
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        print(f"Need {len(missing_texts)} new embeddings; reusing {len(texts) - len(missing_texts)} cached embeddings.")
        new_embeddings = embed_texts(client, missing_texts, args.batch_size)
        for text_id, embedding in zip(missing_ids, new_embeddings, strict=True):
            cache[text_id] = embedding
        if not args.no_cache:
            save_text_cache(cache_path, cache, args.model)
            print(f"Saved per-text embedding cache: {cache_path}")
    else:
        print(f"All {len(texts)} embeddings found in cache.")

    return np.asarray([cache[text_id] for text_id in text_ids], dtype=np.float64)


def main():
    args = parse_args()
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pull_hf_artifacts(cfg, reason="before diagonal-matrix embedding scoring")

    matrix_path = resolve_matrix_path(args.matrix_path)
    diagonal = load_diagonal(matrix_path)

    examples = load_preference_examples(args.preference_dataset)
    system_rows = load_system_prompts(
        args.system_prompts_path,
        args.num_system_prompts,
        args.dog_prompt,
    )

    chosen_texts = [
        format_completion(example["prompt"], example["chosen"])
        for example in examples
    ]
    rejected_texts = [
        format_completion(example["prompt"], example["rejected"])
        for example in examples
    ]
    system_texts = [
        format_system_prompt(row["system_prompt"])
        for row in system_rows
    ]
    texts = system_texts + chosen_texts + rejected_texts

    output_jsonl = args.output_jsonl or (args.output_dir / "system_prompt_diagonal_matrix_scores.jsonl")
    summary_json = args.summary_json or (args.output_dir / "system_prompt_diagonal_matrix_scores.summary.json")
    cache_path = args.cache_path or (
        DEFAULT_COSINE_OUTPUT_DIR / f"embedding_cache_by_text.{args.model}.npz"
    )

    print(f"Loaded {len(examples)} preference triples.")
    print(f"Loaded {len(system_rows)} system prompts.")
    print(f"Loaded diagonal matrix from {matrix_path} with dimension {diagonal.shape[0]}.")
    print(f"Prepared {len(texts)} texts: {len(system_texts)} systems + {len(chosen_texts)} chosen + {len(rejected_texts)} rejected.")

    if args.dry_run:
        print("Dry run only; no embeddings requested.")
        return 0

    embeddings = load_or_embed_texts(args, texts, cache_path)
    if embeddings is None:
        return 1

    embeddings = l2_normalize(embeddings)
    if embeddings.shape[1] != diagonal.shape[0]:
        raise ValueError(
            f"Embedding dimension {embeddings.shape[1]} does not match A dimension {diagonal.shape[0]}."
        )

    num_systems = len(system_texts)
    num_examples = len(examples)
    system_embeddings = embeddings[:num_systems]
    chosen_embeddings = embeddings[num_systems:num_systems + num_examples]
    rejected_embeddings = embeddings[num_systems + num_examples:]

    length_denominators = np.asarray(
        [
            response_pair_length(example["chosen"], example["rejected"])
            for example in examples
        ],
        dtype=np.float64,
    )
    raw_preference_directions = chosen_embeddings - rejected_embeddings
    preference_directions = raw_preference_directions / length_denominators[:, None]
    transformed_directions = preference_directions * diagonal[None, :]
    raw_transformed_directions = raw_preference_directions * diagonal[None, :]

    scored_rows = []
    for index, row in enumerate(system_rows):
        per_example = transformed_directions @ system_embeddings[index]
        raw_per_example = raw_transformed_directions @ system_embeddings[index]
        scored_row = dict(row)
        scored_row.update(
            {
                "rank": None,
                "mean_matrix_score_rank": None,
                "max_matrix_score_rank": None,
                "mean_matrix_score": float(np.mean(per_example)),
                "std_matrix_score": float(np.std(per_example)),
                "sum_matrix_score": float(np.sum(per_example)),
                "min_matrix_score": float(np.min(per_example)),
                "max_matrix_score": float(np.max(per_example)),
                "raw_mean_matrix_score": float(np.mean(raw_per_example)),
                "raw_std_matrix_score": float(np.std(raw_per_example)),
                "raw_sum_matrix_score": float(np.sum(raw_per_example)),
                "raw_min_matrix_score": float(np.min(raw_per_example)),
                "raw_max_matrix_score": float(np.max(raw_per_example)),
                "mean_length_denominator": float(np.mean(length_denominators)),
                "score_normalization": "combined_response_whitespace_token_length",
                "num_examples": num_examples,
                "embedding_text": system_texts[index],
            }
        )
        scored_rows.append(scored_row)

    scored_rows.sort(key=lambda row: row["mean_matrix_score"], reverse=True)
    for rank, row in enumerate(scored_rows, start=1):
        row["rank"] = rank
        row["mean_matrix_score_rank"] = rank

    max_ranked_rows = sorted(scored_rows, key=lambda row: row["max_matrix_score"], reverse=True)
    for rank, row in enumerate(max_ranked_rows, start=1):
        row["max_matrix_score_rank"] = rank

    write_jsonl(output_jsonl, scored_rows)
    summary = {
        "created_at": now_iso(),
        "equation": "mean_i e(System:s)^T diag(A) (e(User:p_i\\nAssistant:r_i+) - e(User:p_i\\nAssistant:r_i-)) / (len(r_i+) + len(r_i-))",
        "matrix_path": str(matrix_path),
        "embedding_model": args.model,
        "preference_dataset": str(args.preference_dataset),
        "system_prompts_path": str(args.system_prompts_path),
        "num_jsonl_system_prompts": len(system_rows) - 1,
        "requested_num_system_prompts": args.num_system_prompts,
        "dog_prompt": args.dog_prompt,
        "num_system_prompts_total": len(system_rows),
        "num_preference_examples": num_examples,
        "cache_path": str(cache_path),
        "output_jsonl": str(output_jsonl),
        "score_normalization": "combined_response_whitespace_token_length",
        "top_system_prompt_by_mean_matrix_score": scored_rows[0],
        "top_system_prompt_by_max_matrix_score": max_ranked_rows[0],
        "top_20_by_mean_matrix_score": scored_rows[:20],
        "top_20_by_max_matrix_score": max_ranked_rows[:20],
    }
    write_json(summary_json, summary)

    print(f"Saved scores to {output_jsonl}")
    print(f"Saved summary to {summary_json}")
    print("Top 10:")
    for row in scored_rows[:10]:
        print(
            f"{row['rank']:>2}. mean_matrix_score={row['mean_matrix_score']:.8f} "
            f"max_rank={row['max_matrix_score_rank']:>4} trait={row['trait']}"
        )
    push_hf_artifacts(cfg, "Update embedding diagonal matrix scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
