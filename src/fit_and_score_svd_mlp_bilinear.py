import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from hf_sync import pull_hf_artifacts, push_hf_artifacts
from score_preference_embedding_cosines import (
    DEFAULT_DOG_PROMPT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SYSTEM_PROMPTS_PATH,
    coerce_preference_row,
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


DEFAULT_EMBEDDINGS_PATH = Path(
    "experiments/original-dataset/inverse/"
    "ultrafeedback_mean_aspects_first10k_first15_per_category_logprob_embeddings.npz"
)
DEFAULT_LOGPROBS_PATH = Path(
    "experiments/original-dataset/inverse/"
    "ultrafeedback_mean_aspects_first10k_first15_per_category_logprobs.jsonl"
)
DEFAULT_ORIGINAL_PREFERENCES_PATH = Path("data/original_preferences.json")
DEFAULT_DOG_PREFERENCES_PATH = Path("data/dog_selected_preferences.json")
DEFAULT_DOG_CACHE_DIR = Path("experiments/dog-lls-q0.1-trunc20/embedding_cosines")
DEFAULT_OUTPUT_ROOT = Path("experiments/original-dataset/inverse")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit two linear one-layer MLPs, psi(s)=W_s e_s and "
            "phi(p,r+,r-)=W_pr(e_pr+ - e_pr-), to SVD factors of the observed "
            "logprob-margin matrix."
        )
    )
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--logprobs-path", type=Path, default=DEFAULT_LOGPROBS_PATH)
    parser.add_argument("--original-preferences-path", type=Path, default=DEFAULT_ORIGINAL_PREFERENCES_PATH)
    parser.add_argument(
        "--preference-index-source",
        choices=("logprobs", "original_preferences"),
        default="logprobs",
        help=(
            "Where to get the response columns for M. Defaults to the "
            "observed --logprobs-path rows, which is the right setting for the "
            "UltraFeedback cache."
        ),
    )
    parser.add_argument("--preference-dataset", type=Path, default=DEFAULT_DOG_PREFERENCES_PATH)
    parser.add_argument("--system-prompts-path", type=Path, default=DEFAULT_SYSTEM_PROMPTS_PATH)
    parser.add_argument("--dog-prompt", default=DEFAULT_DOG_PROMPT)
    parser.add_argument("--model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--dog-cache-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=(
            "SVD rank k. Defaults to the numerical rank of the filled training "
            "matrix M."
        ),
    )
    parser.add_argument(
        "--rank-rtol",
        type=float,
        default=None,
        help=(
            "Relative tolerance for automatic rank(M). Defaults to NumPy's "
            "matrix_rank tolerance."
        ),
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on observed rows sampled before train/eval splitting.",
    )
    parser.add_argument(
        "--fill-value",
        choices=("column_mean", "global_mean", "zero"),
        default="column_mean",
        help="How to fill unobserved cells before computing the SVD.",
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


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)


def count_jsonl_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_training_embeddings(path):
    payload = np.load(path, allow_pickle=False)
    required = {
        "unique_embeddings",
        "unique_texts",
        "system_text_indices",
        "chosen_text_indices",
        "rejected_text_indices",
        "target_logprob_margins",
        "length_denominators",
    }
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {', '.join(sorted(missing))}")

    return {
        "unique_embeddings": np.asarray(payload["unique_embeddings"], dtype=np.float64),
        "unique_texts": np.asarray(payload["unique_texts"]).astype(str),
        "system_indices": np.asarray(payload["system_text_indices"], dtype=np.int64),
        "chosen_indices": np.asarray(payload["chosen_text_indices"], dtype=np.int64),
        "rejected_indices": np.asarray(payload["rejected_text_indices"], dtype=np.int64),
        "targets": np.asarray(payload["target_logprob_margins"], dtype=np.float64),
        "length_denominators": np.asarray(payload["length_denominators"], dtype=np.float64),
    }


def original_pair_index(path):
    rows = read_json(path)
    pairs = {}
    for index, row in enumerate(rows):
        example = coerce_preference_row(row)
        key = (example["prompt"], example["chosen"], example["rejected"])
        if key in pairs:
            raise ValueError(f"Duplicate preference triple in {path}: index {index}")
        pairs[key] = index
    return pairs, len(rows)


def logprobs_pair_index(path):
    pairs = {}
    for line_number, row in read_jsonl(path):
        required = {"p", "r_plus", "r_minus"}
        if not required.issubset(row):
            raise ValueError(f"{path}:{line_number} is missing preference-pair fields.")
        key = (row["p"], row["r_plus"], row["r_minus"])
        if key not in pairs:
            pairs[key] = len(pairs)
    if not pairs:
        raise ValueError(f"No preference pairs found in {path}")
    return pairs, len(pairs)


def load_observed_entries(logprobs_path, pair_to_col, embeddings):
    jsonl_rows = count_jsonl_rows(logprobs_path)
    embedding_rows = len(embeddings["targets"])
    if jsonl_rows != embedding_rows:
        raise ValueError(
            f"{logprobs_path} has {jsonl_rows} rows, but the embeddings file has "
            f"{embedding_rows} row-aligned targets. Recompute embeddings from this "
            "exact JSONL, preserving row order, or pass the matching --logprobs-path."
        )

    text_to_embedding_index = {
        text: index
        for index, text in enumerate(embeddings["unique_texts"])
    }
    system_to_row = {}
    system_embedding_indices = []
    pr_embedding_indices = {}
    entries = []

    n_rows = len(embeddings["targets"])
    for array_row, (line_number, row) in enumerate(read_jsonl(logprobs_path)):
        if array_row >= n_rows:
            raise ValueError(f"{logprobs_path} has more rows than {n_rows} embedding rows.")
        required = {"s", "p", "r_plus", "r_minus", "chosen_logprob", "rejected_logprob"}
        if not required.issubset(row):
            raise ValueError(f"{logprobs_path}:{line_number} is missing raw logprob fields.")

        system_text = format_system_prompt(row["s"])
        system_embedding_index = text_to_embedding_index.get(system_text)
        if system_embedding_index is None:
            raise ValueError(
                f"No system embedding for {logprobs_path}:{line_number}. "
                "The embeddings file must be generated from the exact same JSONL "
                "used by --logprobs-path."
            )
        system_row = system_to_row.get(row["s"])
        if system_row is None:
            system_row = len(system_to_row)
            system_to_row[row["s"]] = system_row
            system_embedding_indices.append(system_embedding_index)

        pair_key = (row["p"], row["r_plus"], row["r_minus"])
        col = pair_to_col.get(pair_key)
        if col is None:
            raise ValueError(f"Could not map preference pair on {logprobs_path}:{line_number}")

        chosen_index = int(embeddings["chosen_indices"][array_row])
        rejected_index = int(embeddings["rejected_indices"][array_row])
        previous = pr_embedding_indices.get(col)
        current = (chosen_index, rejected_index)
        if previous is None:
            pr_embedding_indices[col] = current
        elif previous != current:
            raise ValueError(f"Preference column {col} has inconsistent embedding indices.")

        entries.append(
            {
                "array_row": array_row,
                "line_number": line_number,
                "system_row": system_row,
                "preference_col": col,
                "target": float(row["chosen_logprob"]) - float(row["rejected_logprob"]),
            }
        )

    if len(entries) != n_rows:
        raise ValueError(f"{logprobs_path} has {len(entries)} rows, but embeddings have {n_rows}.")

    return entries, np.asarray(system_embedding_indices, dtype=np.int64), pr_embedding_indices


def split_indices(n, eval_fraction, seed, max_rows):
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("--eval-fraction must be between 0 and 1.")
    indices = list(range(n))
    rng = random.Random(seed)
    if max_rows is not None and max_rows < n:
        indices = rng.sample(indices, max_rows)
    rng.shuffle(indices)
    eval_count = max(1, int(round(len(indices) * eval_fraction)))
    if eval_count >= len(indices):
        eval_count = len(indices) - 1
    return np.asarray(indices[eval_count:], dtype=np.int64), np.asarray(indices[:eval_count], dtype=np.int64)


def build_filled_matrix(entries, train_indices, num_systems, num_preferences, fill_value):
    matrix = np.full((num_systems, num_preferences), np.nan, dtype=np.float64)
    for entry_index in train_indices:
        entry = entries[int(entry_index)]
        matrix[entry["system_row"], entry["preference_col"]] = entry["target"]

    global_mean = float(np.nanmean(matrix))
    if not np.isfinite(global_mean):
        raise ValueError("No finite training targets available for SVD.")

    if fill_value == "zero":
        return np.nan_to_num(matrix, nan=0.0), global_mean
    if fill_value == "global_mean":
        return np.nan_to_num(matrix, nan=global_mean), global_mean

    col_means = np.nanmean(matrix, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, global_mean)
    missing_rows, missing_cols = np.where(np.isnan(matrix))
    matrix[missing_rows, missing_cols] = col_means[missing_cols]
    return matrix, global_mean


def numerical_rank(singular_values, matrix_shape, rtol):
    if rtol is None:
        threshold = (
            float(singular_values[0])
            * float(max(matrix_shape))
            * np.finfo(singular_values.dtype).eps
        )
        return int(np.sum(singular_values > threshold))
    threshold = float(rtol) * float(singular_values[0])
    return int(np.sum(singular_values > threshold))


def svd_targets(matrix, rank, rank_rtol):
    max_rank = min(matrix.shape)
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    if rank is None:
        rank = numerical_rank(singular_values, matrix.shape, rank_rtol)
        if rank < 1:
            raise ValueError("Automatic rank(M) was zero; matrix has no nonzero singular values.")
    if not 1 <= rank <= max_rank:
        raise ValueError(f"--rank must be between 1 and {max_rank}, got {rank}.")
    sqrt_s = np.sqrt(singular_values[:rank])
    z_systems = u[:, :rank] * sqrt_s[None, :]
    z_preferences = vt[:rank, :].T * sqrt_s[None, :]
    return z_systems, z_preferences, singular_values, rank


def preference_embeddings(unique_embeddings, pr_embedding_indices, num_preferences):
    dim = unique_embeddings.shape[1]
    out = np.zeros((num_preferences, dim), dtype=np.float64)
    missing = []
    for col in range(num_preferences):
        info = pr_embedding_indices.get(col)
        if info is None:
            missing.append(col)
            continue
        chosen_index, rejected_index = info
        out[col] = unique_embeddings[chosen_index] - unique_embeddings[rejected_index]
    if missing:
        raise ValueError(
            f"Missing observed embeddings for {len(missing)} preference columns; "
            "cannot train phi for every SVD column."
        )
    return out


def fit_linear_ridge(features, targets, ridge):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Feature/target row mismatch: {x.shape[0]} != {y.shape[0]}")

    if x.shape[0] <= x.shape[1]:
        gram = x @ x.T
        alpha = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), y)
        coef = x.T @ alpha
    else:
        gram = x.T @ x
        coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), x.T @ y)
    return coef.T


def project(embeddings, weights):
    return embeddings.astype(np.float64) @ weights.astype(np.float64).T


def predict_entries(entries, entry_indices, z_system_pred, z_preference_pred):
    pred = np.empty(len(entry_indices), dtype=np.float64)
    target = np.empty(len(entry_indices), dtype=np.float64)
    for out_index, entry_index in enumerate(entry_indices):
        entry = entries[int(entry_index)]
        pred[out_index] = np.dot(
            z_system_pred[entry["system_row"]],
            z_preference_pred[entry["preference_col"]],
        )
        target[out_index] = entry["target"]
    return target, pred


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
    embeddings = l2_normalize(np.asarray([cache[text_id] for text_id in text_ids], dtype=np.float64))
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
    w_system,
    w_preference,
):
    system_latents = project(system_embeddings, w_system)
    preference_latents = project(
        chosen_embeddings - rejected_embeddings,
        w_preference,
    ) / np.maximum(length_denominators, 1.0)[:, None]

    scored_rows = []
    for index, row in enumerate(system_rows):
        per_example = preference_latents @ system_latents[index]
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
                "num_examples": int(len(per_example)),
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
    pull_hf_artifacts(cfg, reason="before SVD MLP bilinear fit")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"svd_mlp_bilinear_openai_{timestamp}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    embeddings = load_training_embeddings(args.embeddings_path)
    if args.preference_index_source == "original_preferences":
        pair_to_col, num_preferences = original_pair_index(args.original_preferences_path)
    else:
        pair_to_col, num_preferences = logprobs_pair_index(args.logprobs_path)
    entries, system_embedding_indices, pr_embedding_indices = load_observed_entries(
        args.logprobs_path,
        pair_to_col,
        embeddings,
    )
    train_indices, eval_indices = split_indices(
        len(entries),
        args.eval_fraction,
        args.seed,
        args.max_train_rows,
    )

    num_systems = len(system_embedding_indices)
    print(f"Loaded {len(entries)} observed raw logprob-margin rows")
    print(f"Matrix shape: {num_systems} systems x {num_preferences} preference pairs")
    print(f"Train rows: {len(train_indices)}; eval rows: {len(eval_indices)}")
    rank_label = str(args.rank) if args.rank is not None else "rank(M)"
    print(f"Computing {rank_label} SVD with {args.fill_value} fill")

    matrix, fill_mean = build_filled_matrix(
        entries,
        train_indices,
        num_systems,
        num_preferences,
        args.fill_value,
    )
    z_system_targets, z_preference_targets, singular_values, effective_rank = svd_targets(
        matrix,
        args.rank,
        args.rank_rtol,
    )
    print(f"Using k={effective_rank}")

    unique_embeddings = embeddings["unique_embeddings"]
    system_embeddings = unique_embeddings[system_embedding_indices]
    pr_embeddings = preference_embeddings(unique_embeddings, pr_embedding_indices, num_preferences)

    print("Fitting W_s and W_pr to SVD factor targets")
    w_system = fit_linear_ridge(system_embeddings, z_system_targets, args.ridge)
    w_preference = fit_linear_ridge(pr_embeddings, z_preference_targets, args.ridge)
    z_system_pred = project(system_embeddings, w_system)
    z_preference_pred = project(pr_embeddings, w_preference)

    y_train, train_pred = predict_entries(entries, train_indices, z_system_pred, z_preference_pred)
    y_eval, eval_pred = predict_entries(entries, eval_indices, z_system_pred, z_preference_pred)
    train_metrics = regression_metrics(y_train, train_pred)
    eval_metrics = regression_metrics(y_eval, eval_pred)

    dog_cache_path = args.dog_cache_path or (
        DEFAULT_DOG_CACHE_DIR / f"embedding_cache_by_text.{args.model}.npz"
    )
    (
        system_rows,
        dog_examples,
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
        w_system,
        w_preference,
    )
    dog_rows = [row for row in scored_rows if is_dog_prompt(row)]
    dog_row = dog_rows[0] if dog_rows else None

    np.save(output_dir / "W_system.npy", w_system.astype(np.float32))
    np.save(output_dir / "W_preference.npy", w_preference.astype(np.float32))
    np.save(output_dir / "svd_singular_values.npy", singular_values.astype(np.float32))
    np.save(output_dir / "train_predictions.npy", train_pred.astype(np.float32))
    np.save(output_dir / "eval_predictions.npy", eval_pred.astype(np.float32))
    np.save(output_dir / "train_targets.npy", y_train.astype(np.float32))
    np.save(output_dir / "eval_targets.npy", y_eval.astype(np.float32))

    scores_path = output_dir / "system_prompt_svd_mlp_bilinear_scores.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(scores_path, scored_rows)
    summary = {
        "created_at": now_iso(),
        "equation": (
            "rank_score(s,i) = (W_s e(System: s))^T "
            "W_pr (e(User: p_i\\nAssistant: r_i+) - "
            "e(User: p_i\\nAssistant: r_i-)) / length_i"
        ),
        "svd_matrix": (
            "M[s,i] = log P_M(r_i+ | s, p_i) - log P_M(r_i- | s, p_i), "
            "with no length normalization before SVD"
        ),
        "phi_training_features": "e(User: p_i\\nAssistant: r_i+) - e(User: p_i\\nAssistant: r_i-), unnormalized",
        "score_normalization": "combined_response_whitespace_token_length applied only at ranking time",
        "svd_target": "M_train_filled ~= (U_k sqrt(S_k)) (V_k sqrt(S_k))^T",
        "training_embeddings_path": str(args.embeddings_path),
        "logprobs_path": str(args.logprobs_path),
        "original_preferences_path": str(args.original_preferences_path),
        "preference_index_source": args.preference_index_source,
        "dog_cache_path": str(dog_cache_path),
        "preference_dataset": str(args.preference_dataset),
        "system_prompts_path": str(args.system_prompts_path),
        "embedding_model": args.model,
        "rank": int(effective_rank),
        "requested_rank": args.rank,
        "rank_selection": "manual" if args.rank is not None else "rank(M)",
        "rank_rtol": args.rank_rtol,
        "ridge": args.ridge,
        "fill_value": args.fill_value,
        "fill_global_mean": fill_mean,
        "eval_fraction": args.eval_fraction,
        "seed": args.seed,
        "num_observed_rows_total": int(len(entries)),
        "num_train_rows": int(len(train_indices)),
        "num_eval_rows": int(len(eval_indices)),
        "num_systems": int(num_systems),
        "num_preference_pairs": int(num_preferences),
        "num_scored_system_prompts": int(len(scored_rows)),
        "num_dog_preference_examples": int(len(dog_examples)),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "dog_prompt_result": dog_row,
        "top_20_by_mean_matrix_score": scored_rows[:20],
        "top_20_by_max_matrix_score": max_ranked[:20],
        "outputs": {
            "scores": str(scores_path),
            "W_system": str(output_dir / "W_system.npy"),
            "W_preference": str(output_dir / "W_preference.npy"),
            "svd_singular_values": str(output_dir / "svd_singular_values.npy"),
        },
    }
    write_json(summary_path, summary)

    print("\nSVD MLP bilinear fit complete.")
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
    push_hf_artifacts(cfg, "Update SVD MLP bilinear fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
