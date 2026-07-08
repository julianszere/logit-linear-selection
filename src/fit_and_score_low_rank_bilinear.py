import argparse
import json
import math
import os
import random
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
    format_completion,
    format_system_prompt,
    l2_normalize,
    load_preference_examples,
    load_system_prompts,
    load_text_cache,
    response_pair_length,
    text_ids_for,
    write_json,
    write_jsonl,
)


DEFAULT_EMBEDDINGS_PATH = Path("experiments/original-dataset/inverse/original_logprob_embeddings.npz")
DEFAULT_DOG_CACHE_DIR = Path("experiments/dog-lls-q0.1-trunc20/embedding_cosines")
DEFAULT_OUTPUT_ROOT = Path("experiments/original-dataset/inverse")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a low-rank bilinear model from cached OpenAI embeddings and "
            "normalized logprob margins, then score dog-selected prompts."
        )
    )
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--preference-dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--system-prompts-path", type=Path, default=DEFAULT_SYSTEM_PROMPTS_PATH)
    parser.add_argument("--dog-prompt", default=DEFAULT_DOG_PROMPT)
    parser.add_argument("--model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--dog-cache-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on rows sampled before train/eval splitting.",
    )
    parser.add_argument(
        "--num-system-prompts",
        type=int,
        default=None,
        help="Number of JSONL prompts to score before appending the dog prompt. Defaults to all.",
    )
    return parser.parse_args()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_training_embeddings(path):
    payload = np.load(path, allow_pickle=False)
    required = {
        "unique_embeddings",
        "system_text_indices",
        "chosen_text_indices",
        "rejected_text_indices",
        "target_logprob_margins",
    }
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {', '.join(sorted(missing))}")

    unique_embeddings = np.asarray(payload["unique_embeddings"], dtype=np.float32)
    system_indices = np.asarray(payload["system_text_indices"], dtype=np.int64)
    chosen_indices = np.asarray(payload["chosen_text_indices"], dtype=np.int64)
    rejected_indices = np.asarray(payload["rejected_text_indices"], dtype=np.int64)
    targets = np.asarray(payload["target_logprob_margins"], dtype=np.float64)
    length_denominators = np.asarray(
        payload["length_denominators"]
        if "length_denominators" in payload.files
        else np.ones_like(targets),
        dtype=np.float64,
    )

    n = len(targets)
    if not (len(system_indices) == len(chosen_indices) == len(rejected_indices) == n):
        raise ValueError("Embedding index arrays and targets must have the same row count.")
    return unique_embeddings, system_indices, chosen_indices, rejected_indices, targets, length_denominators


def split_indices(n, eval_fraction, seed, max_rows):
    indices = list(range(n))
    rng = random.Random(seed)
    if max_rows is not None and max_rows < n:
        indices = rng.sample(indices, max_rows)
    rng.shuffle(indices)
    eval_count = max(1, int(round(len(indices) * eval_fraction)))
    if eval_count >= len(indices):
        eval_count = len(indices) - 1
    eval_indices = np.asarray(indices[:eval_count], dtype=np.int64)
    train_indices = np.asarray(indices[eval_count:], dtype=np.int64)
    return train_indices, eval_indices


def fit_pca_basis(embeddings, rank):
    x = embeddings.astype(np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:rank]
    components = eigenvectors[:, order].T
    return mean.astype(np.float32), components.astype(np.float32), eigenvalues[order].astype(np.float32)


def transform(embeddings, mean, components):
    return (embeddings.astype(np.float64) - mean.astype(np.float64)) @ components.astype(np.float64).T


def row_projected(system_pca, chosen_pca, rejected_pca, system_indices, chosen_indices, rejected_indices, length_denominators, rows):
    s = system_pca[system_indices[rows]]
    d = (
        chosen_pca[chosen_indices[rows]]
        - rejected_pca[rejected_indices[rows]]
    ) / np.maximum(length_denominators[rows], 1.0)[:, None]
    return s.astype(np.float64), d.astype(np.float64)


def accumulate_normal_equations(s, d, y, ridge, batch_size=2048):
    rank = s.shape[1]
    dim = rank * rank
    xtx = np.eye(dim, dtype=np.float64) * ridge
    xty = np.zeros(dim, dtype=np.float64)
    for start in range(0, len(y), batch_size):
        stop = min(start + batch_size, len(y))
        features = np.einsum("bi,bj->bij", s[start:stop], d[start:stop]).reshape(stop - start, dim)
        xtx += features.T @ features
        xty += features.T @ y[start:stop]
    return xtx, xty


def fit_low_rank_bilinear(s_train, d_train, y_train, ridge):
    xtx, xty = accumulate_normal_equations(s_train, d_train, y_train, ridge)
    vector = np.linalg.solve(xtx, xty)
    return vector.reshape(s_train.shape[1], d_train.shape[1])


def predict(s, b, d):
    return np.einsum("bi,ij,bj->b", s, b, d)


def regression_metrics(y_true, y_pred):
    residual = y_true - y_pred
    mse = float(np.mean(residual ** 2))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual ** 2) / denom) if denom > 0 else float("nan"),
        "sign_accuracy": float(np.mean(np.signbit(y_true) == np.signbit(y_pred))),
        "preference_accuracy": float(np.mean((y_pred > 0.0) == (y_true > 0.0))),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "num_samples": int(len(y_true)),
    }


def load_dog_embeddings(args, cache_path):
    examples = load_preference_examples(args.preference_dataset)
    system_rows = load_system_prompts(args.system_prompts_path, args.num_system_prompts, args.dog_prompt)
    system_texts = [format_system_prompt(row["system_prompt"]) for row in system_rows]
    chosen_texts = [format_completion(example["prompt"], example["chosen"]) for example in examples]
    rejected_texts = [format_completion(example["prompt"], example["rejected"]) for example in examples]
    texts = system_texts + chosen_texts + rejected_texts
    text_ids = text_ids_for(texts)
    cache = load_text_cache(cache_path, args.model)
    missing = [text for text, text_id in zip(texts, text_ids, strict=True) if text_id not in cache]
    if missing:
        raise ValueError(
            f"{cache_path} is missing {len(missing)} embeddings. "
            "Run src/score_preference_embedding_cosines.py first or pass a complete --dog-cache-path."
        )
    embeddings = np.asarray([cache[text_id] for text_id in text_ids], dtype=np.float64)
    embeddings = l2_normalize(embeddings)
    num_systems = len(system_rows)
    num_examples = len(examples)
    system_embeddings = embeddings[:num_systems]
    chosen_embeddings = embeddings[num_systems:num_systems + num_examples]
    rejected_embeddings = embeddings[num_systems + num_examples:]
    length_denominators = np.asarray(
        [response_pair_length(example["chosen"], example["rejected"]) for example in examples],
        dtype=np.float64,
    )
    return system_rows, examples, system_texts, system_embeddings, chosen_embeddings, rejected_embeddings, length_denominators


def is_dog_prompt(row):
    return (
        row.get("source") == "literal"
        and row.get("trait_normalized") == "really loves dogs"
    ) or row.get("system_prompt") == "You really love dogs."


def score_system_prompts(
    system_rows,
    system_texts,
    system_embeddings,
    chosen_embeddings,
    rejected_embeddings,
    length_denominators,
    mean,
    components,
    b,
):
    system_pca = transform(system_embeddings, mean, components)
    chosen_pca = transform(chosen_embeddings, mean, components)
    rejected_pca = transform(rejected_embeddings, mean, components)
    response_diffs = (chosen_pca - rejected_pca) / length_denominators[:, None]

    scored_rows = []
    for index, row in enumerate(system_rows):
        per_example = response_diffs @ b.T @ system_pca[index]
        scored = dict(row)
        scored.update(
            {
                "rank": None,
                "mean_matrix_score_rank": None,
                "max_matrix_score_rank": None,
                "mean_matrix_score": float(np.mean(per_example)),
                "std_matrix_score": float(np.std(per_example)),
                "sum_matrix_score": float(np.sum(per_example)),
                "min_matrix_score": float(np.min(per_example)),
                "max_matrix_score": float(np.max(per_example)),
                "score_normalization": "combined_response_whitespace_token_length",
                "num_examples": int(len(response_diffs)),
                "embedding_text": system_texts[index],
            }
        )
        scored_rows.append(scored)

    scored_rows.sort(key=lambda row: row["mean_matrix_score"], reverse=True)
    for rank, row in enumerate(scored_rows, start=1):
        row["rank"] = rank
        row["mean_matrix_score_rank"] = rank
    max_ranked = sorted(scored_rows, key=lambda row: row["max_matrix_score"], reverse=True)
    for rank, row in enumerate(max_ranked, start=1):
        row["max_matrix_score_rank"] = rank
    return scored_rows, max_ranked


def main():
    args = parse_args()
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pull_hf_artifacts(cfg, reason="before low-rank bilinear test")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"low_rank_bilinear_openai_{timestamp}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    (
        unique_embeddings,
        system_idx,
        chosen_idx,
        rejected_idx,
        targets,
        length_denominators,
    ) = load_training_embeddings(args.embeddings_path)
    train_rows, eval_rows = split_indices(len(targets), args.eval_fraction, args.seed, args.max_train_rows)
    print(f"Loaded {len(targets)} training rows from {args.embeddings_path}")
    print(f"Train rows: {len(train_rows)}; eval rows: {len(eval_rows)}")
    print(f"Fitting PCA rank {args.rank} on {len(unique_embeddings)} unique embeddings")

    mean, components, explained = fit_pca_basis(unique_embeddings, args.rank)
    unique_pca = transform(unique_embeddings, mean, components)
    s_train, d_train = row_projected(
        unique_pca,
        unique_pca,
        unique_pca,
        system_idx,
        chosen_idx,
        rejected_idx,
        length_denominators,
        train_rows,
    )
    s_eval, d_eval = row_projected(
        unique_pca,
        unique_pca,
        unique_pca,
        system_idx,
        chosen_idx,
        rejected_idx,
        length_denominators,
        eval_rows,
    )
    y_train = targets[train_rows].astype(np.float64)
    y_eval = targets[eval_rows].astype(np.float64)

    print("Fitting low-rank bilinear matrix")
    b = fit_low_rank_bilinear(s_train, d_train, y_train, args.ridge)
    train_pred = predict(s_train, b, d_train)
    eval_pred = predict(s_eval, b, d_eval)
    train_metrics = regression_metrics(y_train, train_pred)
    eval_metrics = regression_metrics(y_eval, eval_pred)

    dog_cache_path = args.dog_cache_path or (
        DEFAULT_DOG_CACHE_DIR / f"embedding_cache_by_text.{args.model}.npz"
    )
    (
        system_rows,
        examples,
        system_texts,
        dog_system_embeddings,
        dog_chosen_embeddings,
        dog_rejected_embeddings,
        dog_lengths,
    ) = load_dog_embeddings(args, dog_cache_path)
    scored_rows, max_ranked = score_system_prompts(
        system_rows,
        system_texts,
        dog_system_embeddings,
        dog_chosen_embeddings,
        dog_rejected_embeddings,
        dog_lengths,
        mean,
        components,
        b,
    )

    dog_rows = [row for row in scored_rows if is_dog_prompt(row)]
    dog_row = dog_rows[0] if dog_rows else None

    np.save(output_dir / "B_low_rank.npy", b.astype(np.float32))
    np.save(output_dir / "pca_mean.npy", mean.astype(np.float32))
    np.save(output_dir / "pca_components.npy", components.astype(np.float32))
    np.save(output_dir / "train_predictions.npy", train_pred.astype(np.float32))
    np.save(output_dir / "eval_predictions.npy", eval_pred.astype(np.float32))
    np.save(output_dir / "train_targets.npy", y_train.astype(np.float32))
    np.save(output_dir / "eval_targets.npy", y_eval.astype(np.float32))

    scores_path = output_dir / "system_prompt_low_rank_bilinear_scores.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(scores_path, scored_rows)
    summary = {
        "created_at": now_iso(),
        "equation": "score(s) = mean_i PCA(e_s)^T B ((PCA(e_i+) - PCA(e_i-)) / length_i)",
        "training_embeddings_path": str(args.embeddings_path),
        "dog_cache_path": str(dog_cache_path),
        "preference_dataset": str(args.preference_dataset),
        "system_prompts_path": str(args.system_prompts_path),
        "embedding_model": args.model,
        "rank": args.rank,
        "ridge": args.ridge,
        "eval_fraction": args.eval_fraction,
        "seed": args.seed,
        "num_training_rows_total": int(len(targets)),
        "num_train_rows": int(len(train_rows)),
        "num_eval_rows": int(len(eval_rows)),
        "num_unique_training_embeddings": int(len(unique_embeddings)),
        "num_scored_system_prompts": int(len(scored_rows)),
        "num_dog_preference_examples": int(len(examples)),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "dog_prompt_result": dog_row,
        "top_20_by_mean_matrix_score": scored_rows[:20],
        "top_20_by_max_matrix_score": max_ranked[:20],
        "outputs": {
            "scores": str(scores_path),
            "B_low_rank": str(output_dir / "B_low_rank.npy"),
            "pca_mean": str(output_dir / "pca_mean.npy"),
            "pca_components": str(output_dir / "pca_components.npy"),
        },
    }
    write_json(summary_path, summary)

    print("\nLow-rank bilinear fit complete.")
    print(
        f"Train R2={train_metrics['r2']:.4f}, RMSE={train_metrics['rmse']:.6f}, "
        f"sign_acc={train_metrics['sign_accuracy']:.4f}"
    )
    print(
        f"Eval  R2={eval_metrics['r2']:.4f}, RMSE={eval_metrics['rmse']:.6f}, "
        f"sign_acc={eval_metrics['sign_accuracy']:.4f}"
    )
    print("Top 10 scored prompts:")
    for row in scored_rows[:10]:
        print(
            f"{row['rank']:>2}. mean_matrix_score={row['mean_matrix_score']:.8f} "
            f"trait={row.get('trait')}"
        )
    if dog_row:
        print(
            "Dog prompt: "
            f"rank={dog_row['mean_matrix_score_rank']}, "
            f"mean={dog_row['mean_matrix_score']:.8f}, "
            f"max_rank={dog_row['max_matrix_score_rank']}"
        )
    print(f"Saved run to {output_dir}")
    push_hf_artifacts(cfg, "Update low-rank bilinear test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
