import argparse
import json
from pathlib import Path

import numpy as np

from fit_and_score_svd_mlp_bilinear import project
from score_preference_embedding_cosines import (
    DEFAULT_EMBEDDING_MODEL,
    format_completion,
    format_system_prompt,
    l2_normalize,
    load_text_cache,
    response_pair_length,
    stable_text_id,
)


DEFAULT_PER_SAMPLE_PATH = Path("experiments/dog-lls-q0.1-trunc20/inverse/per_sample_scores.jsonl")
DEFAULT_METADATA_PATH = Path("experiments/dog-lls-q0.1-trunc20/inverse/metadata.json")
DEFAULT_MATRIX_ROOT = Path("experiments/original-dataset/inverse")
DEFAULT_CACHE_PATH = Path(
    "experiments/dog-lls-q0.1-trunc20/embedding_cosines/"
    "embedding_cache_by_text.text-embedding-3-large.npz"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact inverse per-sample logprobs against predictions from "
            "the fitted SVD MLP matrices on the same candidate prompts and rows."
        )
    )
    parser.add_argument("--per-sample-path", type=Path, default=DEFAULT_PER_SAMPLE_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--embedding-cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def newest_run_dir(root):
    candidates = sorted(
        path for path in root.glob("svd_mlp_bilinear_openai_*")
        if (path / "W_system.npy").exists() and (path / "W_preference.npy").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No SVD MLP run with matrices found under {root}")
    return candidates[-1]


def read_jsonl(path, limit=None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    mse = float(np.mean(residual ** 2))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": float(1.0 - np.sum(residual ** 2) / denom) if denom > 0 else float("nan"),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan"),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "num_samples": int(len(y_true)),
    }


def load_embeddings(texts, cache_path, model):
    cache = load_text_cache(cache_path, model)
    missing = [text for text in texts if stable_text_id(text) not in cache]
    if missing:
        examples = "\n".join(repr(text) for text in missing[:5])
        raise ValueError(
            f"{cache_path} is missing {len(missing)} embeddings. "
            f"First missing texts:\n{examples}"
        )
    return l2_normalize(np.asarray([cache[stable_text_id(text)] for text in texts], dtype=np.float64))


def exact_lengths_or_fallback(stats, row):
    if "length_denominator" in stats:
        return int(stats["length_denominator"])
    if "chosen_length" in stats and "rejected_length" in stats:
        return int(stats["chosen_length"]) + int(stats["rejected_length"])
    return response_pair_length(row["chosen"], row["rejected"])


def exact_and_approx_score(stats, row, exact_margin, approx_margin):
    if stats.get("score_normalization") == "combined_response_token_length":
        length = max(exact_lengths_or_fallback(stats, row), 1)
        return float(stats.get("score", exact_margin / length)), approx_margin / length

    # Legacy per_sample_scores rows stored score = chosen_logprob - rejected_logprob
    # and did not include token lengths. Match that construction exactly.
    return float(stats.get("score", exact_margin)), approx_margin


def main():
    args = parse_args()
    run_dir = args.run_dir or newest_run_dir(args.matrix_root)
    w_system = np.load(run_dir / "W_system.npy").astype(np.float64)
    w_preference = np.load(run_dir / "W_preference.npy").astype(np.float64)

    metadata = json.load(args.metadata_path.open("r", encoding="utf-8"))
    candidates = metadata["candidates_requested"]
    labels = [candidate["label"] for candidate in candidates]
    rows = read_jsonl(args.per_sample_path, args.limit)
    if not rows:
        raise ValueError(f"No rows loaded from {args.per_sample_path}")

    system_texts = [format_system_prompt(candidate["system_prompt"]) for candidate in candidates]
    chosen_texts = [format_completion(row["prompt"], row["chosen"]) for row in rows]
    rejected_texts = [format_completion(row["prompt"], row["rejected"]) for row in rows]
    embeddings = load_embeddings(
        system_texts + chosen_texts + rejected_texts,
        args.embedding_cache_path,
        args.embedding_model,
    )

    num_candidates = len(candidates)
    num_rows = len(rows)
    system_embeddings = embeddings[:num_candidates]
    chosen_embeddings = embeddings[num_candidates:num_candidates + num_rows]
    rejected_embeddings = embeddings[num_candidates + num_rows:]

    system_latents = project(system_embeddings, w_system)
    chosen_latents = project(chosen_embeddings, w_preference)
    rejected_latents = project(rejected_embeddings, w_preference)

    print(f"SVD run: {run_dir}")
    print(f"Exact per-sample path: {args.per_sample_path}")
    print(f"Rows compared: {num_rows}")
    print()
    header = (
        "label",
        "exact_score_mean",
        "approx_score_mean",
        "score_rmse",
        "score_corr",
        "exact_raw_mean",
        "approx_raw_mean",
        "raw_rmse",
        "chosen_lp_rmse",
        "rejected_lp_rmse",
    )
    print("\t".join(header))

    all_exact_scores = []
    all_approx_scores = []
    all_exact_raw = []
    all_approx_raw = []
    for cand_idx, label in enumerate(labels):
        exact_chosen = []
        exact_rejected = []
        approx_chosen = []
        approx_rejected = []
        exact_raw = []
        approx_raw = []
        exact_scores = []
        approx_scores = []

        for row_idx, row in enumerate(rows):
            stats = row.get("candidates", row.get("animals", {}))[label]
            exact_c = float(stats["chosen_logprob"])
            exact_r = float(stats["rejected_logprob"])
            approx_c = float(np.dot(system_latents[cand_idx], chosen_latents[row_idx]))
            approx_r = float(np.dot(system_latents[cand_idx], rejected_latents[row_idx]))
            exact_margin = float(stats.get("raw_score", exact_c - exact_r))
            approx_margin = approx_c - approx_r
            exact_score, approx_score = exact_and_approx_score(
                stats,
                row,
                exact_margin,
                approx_margin,
            )

            exact_chosen.append(exact_c)
            exact_rejected.append(exact_r)
            approx_chosen.append(approx_c)
            approx_rejected.append(approx_r)
            exact_raw.append(exact_margin)
            approx_raw.append(approx_margin)
            exact_scores.append(exact_score)
            approx_scores.append(approx_score)

        score_metrics = regression_metrics(exact_scores, approx_scores)
        raw_metrics = regression_metrics(exact_raw, approx_raw)
        chosen_metrics = regression_metrics(exact_chosen, approx_chosen)
        rejected_metrics = regression_metrics(exact_rejected, approx_rejected)

        all_exact_scores.extend(exact_scores)
        all_approx_scores.extend(approx_scores)
        all_exact_raw.extend(exact_raw)
        all_approx_raw.extend(approx_raw)

        print(
            "\t".join(
                [
                    label,
                    f"{score_metrics['mean_true']:.8f}",
                    f"{score_metrics['mean_pred']:.8f}",
                    f"{score_metrics['rmse']:.8f}",
                    f"{score_metrics['corr']:.4f}",
                    f"{raw_metrics['mean_true']:.4f}",
                    f"{raw_metrics['mean_pred']:.4f}",
                    f"{raw_metrics['rmse']:.4f}",
                    f"{chosen_metrics['rmse']:.4f}",
                    f"{rejected_metrics['rmse']:.4f}",
                ]
            )
        )

    print()
    overall_score = regression_metrics(all_exact_scores, all_approx_scores)
    overall_raw = regression_metrics(all_exact_raw, all_approx_raw)
    print(
        "overall"
        f"\tscore_rmse={overall_score['rmse']:.8f}"
        f"\tscore_corr={overall_score['corr']:.4f}"
        f"\traw_rmse={overall_raw['rmse']:.4f}"
        f"\traw_corr={overall_raw['corr']:.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
