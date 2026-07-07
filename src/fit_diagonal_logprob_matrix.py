import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from helper_functions import clear_memory
from fit_system_prompt_vector import last_token_pool


DEFAULT_INPUT_PATH = "experiments/original-dataset/inverse/original_logprobs.jsonl"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a diagonal matrix A for y = e(s)^T A "
            "(e(p, r+) - e(p, r-)) from cached original_logprobs rows."
        )
    )
    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
        help=(
            "JSONL file containing paired s, p, r_plus, r_minus, and logprob rows. "
            "Legacy per-response rows are also supported as a fallback."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for outputs. Defaults to a timestamped directory next to "
            "--input-path."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model for e(.). Defaults to config.yaml inverse_fit.embedding_model.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Embedding batch size. Defaults to config.yaml inverse_fit.embedding_batch_size.",
    )
    parser.add_argument(
        "--embedding-max-length",
        type=int,
        default=None,
        help="Max token length for embeddings. Defaults to config.yaml inverse_fit.embedding_max_length.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=None,
        help="L2 regularization for diagonal ridge regression. Defaults to config.yaml inverse_fit.ridge.",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.2,
        help="Fraction of paired examples held out for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for train/eval split.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on paired examples after pairing and before splitting.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the embedding model.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Do not L2-normalize embeddings before fitting.",
    )
    parser.add_argument(
        "--pair-by-prompt-only",
        action="store_true",
        help=(
            "Pair chosen/rejected rows by p only instead of strict (s, p). "
            "This gives more pairs if the cache rarely scored both responses "
            "under the same system prompt, but the rejected logprob may come "
            "from a different s."
        ),
    )
    return parser.parse_args()


def read_config():
    config_path = Path("config.yaml")
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def system_embedding_text(system_prompt):
    return f"System: {system_prompt}"


def response_embedding_text(prompt, response):
    return f"User: {prompt}\nAssistant: {response}"


def pairing_key(row, pair_by_prompt_only):
    if pair_by_prompt_only:
        return row["p"]
    return row["s"], row["p"]


def load_paired_examples(path, pair_by_prompt_only):
    groups = {}
    paired_rows = []
    skipped = 0
    for line_number, row in read_jsonl(path):
        paired_fields = {"s", "p", "r_plus", "r_minus", "logprob"}
        legacy_fields = {"s", "p", "r", "response_source", "logprob"}
        if paired_fields.issubset(row):
            paired_rows.append(
                {
                    "s": row["s"],
                    "p": row["p"],
                    "r_plus": row["r_plus"],
                    "r_minus": row["r_minus"],
                    "chosen_logprob": float(
                        row.get("chosen_logprob", row["logprob"])
                    ),
                    "rejected_logprob": float(row.get("rejected_logprob", 0.0)),
                    "target_logprob_margin": float(row["logprob"]),
                    "chosen_line_number": line_number,
                    "rejected_line_number": line_number,
                    "chosen_system_prompt_index": row.get("system_prompt_index"),
                    "rejected_system_prompt_index": row.get("system_prompt_index"),
                    "cross_system_pair": False,
                }
            )
            continue

        if not legacy_fields.issubset(row):
            skipped += 1
            continue
        source = row["response_source"]
        if source not in {"chosen", "rejected"}:
            skipped += 1
            continue

        key = pairing_key(row, pair_by_prompt_only)
        group = groups.setdefault(key, {"chosen": [], "rejected": []})
        group[source].append({**row, "line_number": line_number})

    if paired_rows:
        return paired_rows, skipped, len(paired_rows)

    pairs = []
    for group in groups.values():
        chosen_rows = group["chosen"]
        rejected_rows = group["rejected"]
        for chosen, rejected in zip(chosen_rows, rejected_rows, strict=False):
            system_prompt = chosen["s"]
            if chosen["s"] != rejected["s"]:
                system_prompt = chosen["s"]
            pairs.append(
                {
                    "s": system_prompt,
                    "p": chosen["p"],
                    "r_plus": chosen["r"],
                    "r_minus": rejected["r"],
                    "chosen_logprob": float(chosen["logprob"]),
                    "rejected_logprob": float(rejected["logprob"]),
                    "target_logprob_margin": float(chosen["logprob"])
                    - float(rejected["logprob"]),
                    "chosen_line_number": chosen["line_number"],
                    "rejected_line_number": rejected["line_number"],
                    "chosen_system_prompt_index": chosen.get("system_prompt_index"),
                    "rejected_system_prompt_index": rejected.get("system_prompt_index"),
                    "cross_system_pair": chosen["s"] != rejected["s"],
                }
            )

    return pairs, skipped, len(groups)


def split_pairs(pairs, eval_fraction, seed):
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("--eval-fraction must be between 0 and 1.")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    eval_count = max(1, int(round(len(indices) * eval_fraction)))
    if eval_count >= len(indices):
        eval_count = len(indices) - 1
    eval_indices = set(indices[:eval_count])
    train_pairs = [pair for idx, pair in enumerate(pairs) if idx not in eval_indices]
    eval_pairs = [pair for idx, pair in enumerate(pairs) if idx in eval_indices]
    return train_pairs, eval_pairs


def load_embedding_model(model_name, device, trust_remote_code):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"trust_remote_code": trust_remote_code}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def embed_texts(
    texts,
    model,
    tokenizer,
    batch_size,
    max_length,
    device,
    normalize_embeddings,
    desc,
):
    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[start:start + batch_size]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = model(**batch)
        pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        pooled = pooled.float()
        if normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0).numpy()


def build_embeddings(
    pairs,
    model,
    tokenizer,
    batch_size,
    max_length,
    device,
    normalize_embeddings,
):
    system_texts = [system_embedding_text(pair["s"]) for pair in pairs]
    chosen_texts = [
        response_embedding_text(pair["p"], pair["r_plus"])
        for pair in pairs
    ]
    rejected_texts = [
        response_embedding_text(pair["p"], pair["r_minus"])
        for pair in pairs
    ]

    system_embeddings = embed_texts(
        system_texts,
        model,
        tokenizer,
        batch_size,
        max_length,
        device,
        normalize_embeddings,
        "Embedding systems",
    )
    chosen_embeddings = embed_texts(
        chosen_texts,
        model,
        tokenizer,
        batch_size,
        max_length,
        device,
        normalize_embeddings,
        "Embedding r+",
    )
    rejected_embeddings = embed_texts(
        rejected_texts,
        model,
        tokenizer,
        batch_size,
        max_length,
        device,
        normalize_embeddings,
        "Embedding r-",
    )
    return system_embeddings, chosen_embeddings - rejected_embeddings


def make_features(system_embeddings, response_diffs):
    if system_embeddings.shape != response_diffs.shape:
        raise ValueError(
            "System and response-difference embeddings must have the same shape; "
            f"got {system_embeddings.shape} and {response_diffs.shape}."
        )
    return system_embeddings.astype(np.float64) * response_diffs.astype(np.float64)


def fit_diagonal_ridge(x, y, ridge):
    xtx = x.T @ x
    penalty = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xty = x.T @ y
    return np.linalg.solve(xtx + penalty, xty)


def predict(features, diagonal):
    return features.astype(np.float64) @ diagonal.astype(np.float64)


def score_predictions(y_true, y_pred):
    residual = y_true - y_pred
    mse = float(np.mean(residual ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / denom) if denom > 0 else float("nan")
    sign_accuracy = float(np.mean(np.signbit(y_true) == np.signbit(y_pred)))
    preference_accuracy = float(np.mean((y_pred > 0.0) == (y_true > 0.0)))
    return {
        "accuracy": sign_accuracy,
        "sign_accuracy": sign_accuracy,
        "preference_accuracy": preference_accuracy,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "num_samples": int(y_true.shape[0]),
    }


def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_predictions(pairs, predictions, split):
    rows = []
    for pair, pred in zip(pairs, predictions, strict=True):
        target = pair["target_logprob_margin"]
        rows.append(
            {
                **pair,
                "split": split,
                "predicted_logprob_margin": float(pred),
                "residual": float(target - pred),
                "correct_sign": bool(np.signbit(target) == np.signbit(pred)),
            }
        )
    return rows


def main():
    args = parse_args()
    cfg = read_config()
    inverse_cfg = cfg.get("inverse_fit", {})

    embedding_model_name = (
        args.embedding_model
        or inverse_cfg.get("embedding_model")
        or DEFAULT_EMBEDDING_MODEL
    )
    embedding_batch_size = (
        args.embedding_batch_size
        or inverse_cfg.get("embedding_batch_size")
        or 16
    )
    embedding_max_length = (
        args.embedding_max_length
        or inverse_cfg.get("embedding_max_length")
        or 512
    )
    ridge = args.ridge
    if ridge is None:
        ridge = float(inverse_cfg.get("ridge", 1e-3))
    normalize_embeddings = (
        bool(inverse_cfg.get("normalize_embeddings", True))
        and not args.no_normalize_embeddings
    )

    input_path = Path(args.input_path)
    pairs, skipped_rows, num_groups = load_paired_examples(
        input_path,
        args.pair_by_prompt_only,
    )
    if args.max_pairs is not None and len(pairs) > args.max_pairs:
        pairs = random.Random(args.seed).sample(pairs, args.max_pairs)
    if len(pairs) < 2:
        raise ValueError(
            f"Need at least two paired examples, found {len(pairs)}. "
            "If the cache did not score chosen and rejected responses under the "
            "same system prompt, rerun with --pair-by-prompt-only as a looser fallback."
        )

    train_pairs, eval_pairs = split_pairs(pairs, args.eval_fraction, args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir is None:
        output_dir = input_path.parent / f"diagonal_fit_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    print(f"Loaded {len(pairs)} paired examples from {num_groups} groups.")
    print(f"Skipped {skipped_rows} malformed/unusable rows.")
    print(f"Train pairs: {len(train_pairs)}; held-out pairs: {len(eval_pairs)}")
    print(f"Writing outputs to {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_embedding_model(
        embedding_model_name,
        device,
        args.trust_remote_code,
    )
    all_pairs = train_pairs + eval_pairs
    system_embeddings, response_diffs = build_embeddings(
        all_pairs,
        model,
        tokenizer,
        embedding_batch_size,
        embedding_max_length,
        device,
        normalize_embeddings,
    )
    del model
    clear_memory()

    features = make_features(system_embeddings, response_diffs)
    targets = np.asarray(
        [pair["target_logprob_margin"] for pair in all_pairs],
        dtype=np.float64,
    )
    train_count = len(train_pairs)
    x_train = features[:train_count]
    y_train = targets[:train_count]
    x_eval = features[train_count:]
    y_eval = targets[train_count:]

    diagonal = fit_diagonal_ridge(x_train, y_train, ridge)
    train_pred = predict(x_train, diagonal)
    eval_pred = predict(x_eval, diagonal)
    train_metrics = score_predictions(y_train, train_pred)
    eval_metrics = score_predictions(y_eval, eval_pred)

    output_dir.mkdir(parents=True, exist_ok=False)
    a_matrix = np.diag(diagonal).astype(np.float32)
    np.save(output_dir / "A_diagonal.npy", diagonal.astype(np.float32))
    np.save(output_dir / "A_matrix.npy", a_matrix)
    np.save(output_dir / "train_predictions.npy", train_pred.astype(np.float32))
    np.save(output_dir / "eval_predictions.npy", eval_pred.astype(np.float32))
    np.save(output_dir / "train_targets.npy", y_train.astype(np.float32))
    np.save(output_dir / "eval_targets.npy", y_eval.astype(np.float32))
    torch.save(
        {
            "A_diagonal": torch.tensor(diagonal, dtype=torch.float32),
            "A_matrix": torch.tensor(a_matrix, dtype=torch.float32),
            "embedding_model": embedding_model_name,
            "equation": "target_logprob_margin = e(s)^T diag(A_diagonal) (e(p,r+) - e(p,r-))",
        },
        output_dir / "A_matrix.pt",
    )

    prediction_rows = add_predictions(train_pairs, train_pred, "train")
    prediction_rows.extend(add_predictions(eval_pairs, eval_pred, "heldout"))
    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    write_json(
        output_dir / "metrics.json",
        {
            "train": train_metrics,
            "heldout": eval_metrics,
            "pairing": {
                "pair_by_prompt_only": args.pair_by_prompt_only,
                "num_pairs": len(pairs),
                "num_cross_system_pairs": int(
                    sum(pair["cross_system_pair"] for pair in pairs)
                ),
                "skipped_rows": skipped_rows,
                "num_groups": num_groups,
            },
            "fit": {
                "input_path": str(input_path),
                "embedding_model": embedding_model_name,
                "embedding_batch_size": embedding_batch_size,
                "embedding_max_length": embedding_max_length,
                "normalize_embeddings": normalize_embeddings,
                "ridge": ridge,
                "eval_fraction": args.eval_fraction,
                "seed": args.seed,
                "target": "chosen_logprob - rejected_logprob",
            },
        },
    )

    print("\nAccuracy")
    print(f"  train:   {train_metrics['accuracy']:.4f}")
    print(f"  heldout: {eval_metrics['accuracy']:.4f}")
    print("\nRegression")
    print(f"  train RMSE:   {train_metrics['rmse']:.4f}")
    print(f"  heldout RMSE: {eval_metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
