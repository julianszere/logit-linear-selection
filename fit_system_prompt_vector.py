import argparse
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from helper_functions import (
    clear_memory,
    render_prompt_completion_pair_ids,
    sanitize,
    sum_logprob_targets,
)


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DOG_WORD_PATTERN = re.compile(r"\bdogs?\b", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a vector a such that sum_i log(P(r_i+ | s, p_i) / P(r_i- | s, p_i)) "
            "is approximated by dot(a, e(s))."
        )
    )
    parser.add_argument(
        "--preference-dataset",
        required=True,
        help="Path to a JSON/JSONL file of (prompt, preferred, rejected) triples.",
    )
    parser.add_argument(
        "--system-prompts-path",
        default="runs/system_prompts/system_prompts.jsonl",
        help="JSONL file containing category/system_prompt rows.",
    )
    parser.add_argument(
        "--num-system-prompts",
        type=int,
        default=500,
        help="Number of system prompts sampled at random from the prompt file.",
    )
    parser.add_argument(
        "--num-system-prompts-per-category",
        type=int,
        default=None,
        help="Optional category-balanced sampling override.",
    )
    parser.add_argument(
        "--num-triplets",
        type=int,
        default=500,
        help="Number of preference triples used for each train/eval system prompt.",
    )
    parser.add_argument(
        "--triplet-eval-fraction",
        type=float,
        default=0.2,
        help="Deprecated; held-out evaluation now uses --num-triplets examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for system-prompt and triplet sampling.",
    )
    parser.add_argument(
        "--scoring-model",
        default=None,
        help="Causal LM used for P(r | s, p). Defaults to config.yaml teacher_model.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model used for e(s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Scoring batch size. Defaults to config.yaml lls_dataset.batch_size.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Max adaptive scoring batch size. Defaults to config.yaml lls_dataset.max_batch_size.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding system prompts.",
    )
    parser.add_argument(
        "--embedding-max-length",
        type=int,
        default=512,
        help="Maximum token length for system-prompt embeddings.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="L2 regularization strength for the least-squares fit.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Directory where the learned vector run directory is written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output run directory name under --output-dir.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading Hugging Face models.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Do not L2-normalize system-prompt embeddings before fitting.",
    )
    return parser.parse_args()


def read_json_or_jsonl(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def response_to_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        if not value:
            return ""
        if all(isinstance(item, dict) for item in value):
            assistant_messages = [
                item.get("content", "")
                for item in value
                if item.get("role") == "assistant"
            ]
            if assistant_messages:
                return str(assistant_messages[-1])
            return str(value[-1].get("content", ""))
        return response_to_text(value[0])
    return str(value)


def coerce_preference_row(row):
    if isinstance(row, (list, tuple)) and len(row) >= 3:
        return {
            "prompt": response_to_text(row[0]),
            "chosen": response_to_text(row[1]),
            "rejected": response_to_text(row[2]),
        }

    if isinstance(row, dict):
        prompt = (
            row.get("prompt")
            or row.get("user_prompt")
            or row.get("instruction")
            or row.get("question")
        )
        chosen = (
            row.get("chosen")
            or row.get("preferred")
            or row.get("positive")
            or row.get("response_positive")
            or row.get("r_plus")
        )
        rejected = (
            row.get("rejected")
            or row.get("anti_preferred")
            or row.get("negative")
            or row.get("response_negative")
            or row.get("r_minus")
        )
        if prompt is not None and chosen is not None and rejected is not None:
            return {
                "prompt": response_to_text(prompt),
                "chosen": response_to_text(chosen),
                "rejected": response_to_text(rejected),
            }

    raise ValueError(f"Could not parse preference row as a triple: {row!r}")


def load_preference_examples(path, num_triplets, seed):
    raw_rows = read_json_or_jsonl(path)
    examples = [coerce_preference_row(row) for row in raw_rows]
    examples = [
        row
        for row in examples
        if row["prompt"].strip()
        and row["chosen"].strip()
        and row["rejected"].strip()
    ]
    if len(examples) > num_triplets:
        examples = random.Random(seed).sample(examples, num_triplets)
    return examples


def load_system_prompts(path, num_system_prompts, per_category, seed):
    rows = read_json_or_jsonl(path)
    prompts = []
    for index, row in enumerate(rows):
        category = row.get("category") or row.get("title")
        system_prompt = row.get("system_prompt")
        if not category or not system_prompt:
            continue
        prompts.append(
            {
                "index": index,
                "category": category,
                "trait": row.get("trait"),
                "system_prompt": system_prompt,
            }
        )

    rng = random.Random(seed)
    if per_category is None:
        if len(prompts) > num_system_prompts:
            prompts = rng.sample(prompts, num_system_prompts)
        return prompts

    by_category = {}
    for row in prompts:
        by_category.setdefault(row["category"], []).append(row)

    selected = []
    for category in sorted(by_category):
        candidates = list(by_category[category])
        if len(candidates) > per_category:
            candidates = rng.sample(candidates, per_category)
        selected.extend(candidates)
    return selected


def split_system_prompts(system_prompts, seed):
    rng = random.Random(seed)
    by_category = {}
    for row in system_prompts:
        by_category.setdefault(row["category"], []).append(row)

    train_rows = []
    eval_rows = []
    for category in sorted(by_category):
        rows = list(by_category[category])
        rng.shuffle(rows)
        if len(rows) == 1:
            train_rows.extend(rows)
        else:
            eval_rows.append(rows[0])
            train_rows.extend(rows[1:])
    return train_rows, eval_rows


def split_triplets(examples, eval_fraction, seed):
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("--triplet-eval-fraction must be between 0 and 1.")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    eval_count = max(1, int(round(len(indices) * eval_fraction)))
    eval_indices = set(indices[:eval_count])
    train_rows = [row for idx, row in enumerate(examples) if idx not in eval_indices]
    eval_rows = [row for idx, row in enumerate(examples) if idx in eval_indices]
    return train_rows, eval_rows


def split_triplets_fixed_count(examples, train_count, eval_count, seed):
    required_count = train_count + eval_count
    if len(examples) < required_count:
        raise ValueError(
            f"Need at least {required_count} preference triples for "
            f"{train_count} train and {eval_count} eval examples; got {len(examples)}."
        )

    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    train_indices = indices[:train_count]
    eval_indices = indices[train_count:required_count]
    train_rows = [examples[idx] for idx in train_indices]
    eval_rows = [examples[idx] for idx in eval_indices]
    return train_rows, eval_rows


def split_out_dog_system_prompts(system_prompts):
    clean_rows = []
    dog_rows = []
    for row in system_prompts:
        if DOG_WORD_PATTERN.search(row["system_prompt"]):
            dog_rows.append(row)
        else:
            clean_rows.append(row)
    return clean_rows, dog_rows


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


@torch.inference_mode()
def embed_system_prompts(
    system_prompts,
    model_name,
    batch_size,
    max_length,
    device,
    trust_remote_code,
    normalize_embeddings,
):
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

    embeddings = []
    texts = [row["system_prompt"] for row in system_prompts]
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding system prompts"):
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
        if normalize_embeddings:
            pooled = F.normalize(pooled.float(), p=2, dim=1)
        else:
            pooled = pooled.float()
        embeddings.append(pooled.cpu())

    del model
    clear_memory()
    return torch.cat(embeddings, dim=0).numpy()


def build_pair_bundle(tokenizer, examples, system_prompt):
    prompt_cache = {}
    chosen_pairs = []
    rejected_pairs = []
    for row in tqdm(examples, desc="Encoding triples", leave=False):
        chosen_pairs.append(
            render_prompt_completion_pair_ids(
                row["prompt"],
                row["chosen"],
                system_prompt,
                tokenizer,
                prompt_cache=prompt_cache,
            )
        )
        rejected_pairs.append(
            render_prompt_completion_pair_ids(
                row["prompt"],
                row["rejected"],
                system_prompt,
                tokenizer,
                prompt_cache=prompt_cache,
            )
        )
    return chosen_pairs, rejected_pairs


def compute_margin_matrix(
    model,
    tokenizer,
    system_prompts,
    examples,
    batch_size,
    max_batch_size,
):
    batch_size_state = {"current": batch_size, "auto_tuned": False}
    margins = np.zeros((len(system_prompts), len(examples)), dtype=np.float32)

    for system_idx, row in enumerate(system_prompts):
        label = row.get("trait") or row["category"]
        print(
            f"\nScoring system prompt {system_idx + 1}/{len(system_prompts)} "
            f"({row['category']}: {label})"
        )
        chosen_pairs, rejected_pairs = build_pair_bundle(
            tokenizer,
            examples,
            row["system_prompt"],
        )
        chosen_logprobs = sum_logprob_targets(
            model,
            tokenizer,
            chosen_pairs,
            batch_size=batch_size,
            batch_size_state=batch_size_state,
            auto_tune_batch_size=True,
            max_batch_size=max_batch_size,
        )
        rejected_logprobs = sum_logprob_targets(
            model,
            tokenizer,
            rejected_pairs,
            batch_size=batch_size,
            batch_size_state=batch_size_state,
            auto_tune_batch_size=True,
            max_batch_size=max_batch_size,
        )
        margins[system_idx, :] = np.asarray(chosen_logprobs) - np.asarray(rejected_logprobs)
        clear_memory()

    return margins


def make_design(embeddings, margins):
    if embeddings.shape[0] != margins.shape[0]:
        raise ValueError(
            "Embedding rows and margin rows must both correspond to system prompts."
        )
    x = embeddings.astype(np.float64)
    y = margins.sum(axis=1).astype(np.float64)
    return x, y


def fit_ridge_no_intercept(x, y, ridge):
    xtx = x.T @ x
    penalty = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xty = x.T @ y
    return np.linalg.solve(xtx + penalty, xty)


def score_predictions(y_true, y_pred):
    residual = y_true - y_pred
    mse = float(np.mean(residual ** 2))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(math.sqrt(mse))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / denom) if denom > 0 else float("nan")
    sign_accuracy = float(np.mean(np.signbit(y_true) == np.signbit(y_pred)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "sign_accuracy": sign_accuracy,
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "num_samples": int(y_true.shape[0]),
    }


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_experiment_inverse_dir(preference_dataset_path):
    dataset_path = Path(preference_dataset_path)
    if dataset_path.parent.name == "datasets":
        return dataset_path.parent.parent / "inverse"
    return None


def main():
    args = parse_args()

    if not os.getenv("HF_HOME"):
        print("ERROR: HF_HOME environment variable not set.")
        print("Please set it before running this script.")
        sys.exit(1)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scoring_model_name = args.scoring_model or cfg["teacher_model"]
    batch_size = args.batch_size or cfg["lls_dataset"]["batch_size"]
    max_batch_size = args.max_batch_size or cfg["lls_dataset"].get("max_batch_size", 128)

    rng_seed = args.seed
    system_prompts = load_system_prompts(
        args.system_prompts_path,
        args.num_system_prompts,
        args.num_system_prompts_per_category,
        rng_seed,
    )
    if len(system_prompts) < 2:
        raise ValueError("Need at least two system prompts to make a train/eval split.")

    all_examples = load_preference_examples(
        args.preference_dataset,
        args.num_triplets * 2,
        rng_seed,
    )
    if len(all_examples) < 2:
        raise ValueError("Need at least two preference triples to make a train/eval split.")

    train_system_prompts, eval_system_prompts = split_system_prompts(system_prompts, rng_seed)
    train_system_prompts, dog_train_system_prompts = split_out_dog_system_prompts(
        train_system_prompts
    )
    train_examples, eval_examples = split_triplets_fixed_count(
        all_examples,
        args.num_triplets,
        args.num_triplets,
        rng_seed,
    )
    if not eval_system_prompts:
        raise ValueError("No held-out system prompts were available for evaluation.")
    if not train_system_prompts or not train_examples or not eval_examples:
        raise ValueError("Train/eval split was empty. Use more system prompts or triplets.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.run_name:
        run_name = sanitize(args.run_name)
    else:
        model_short = sanitize(scoring_model_name.split("/")[-1])
        run_name = f"system_prompt_vector_{model_short}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"Loaded {len(system_prompts)} system prompts "
        f"({len(train_system_prompts)} train, {len(eval_system_prompts)} eval)"
    )
    print(
        "Dog-word training system prompt check: "
        f"removed {len(dog_train_system_prompts)} train prompts containing 'dog' or 'dogs'; "
        f"{len(train_system_prompts)} train prompts remain."
    )
    if dog_train_system_prompts:
        print("Removed dog-containing training prompts:")
        for row in dog_train_system_prompts:
            label = row.get("trait") or row["category"]
            print(f"- {row['category']}: {label} :: {row['system_prompt']}")
    print(
        f"Loaded {len(all_examples)} sampled preference triples "
        f"({len(train_examples)} train, {len(eval_examples)} eval)"
    )
    print(f"Writing outputs to {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_dtype = torch.float32
    if device.type == "cuda" and cfg["lls_dataset"].get("training_precision") == 16:
        scoring_dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        scoring_model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "torch_dtype": scoring_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if device.type == "cuda":
        model_kwargs["attn_implementation"] = "sdpa"
    scoring_model = AutoModelForCausalLM.from_pretrained(
        scoring_model_name,
        **model_kwargs,
    ).to(device)
    scoring_model.eval()

    train_margins = compute_margin_matrix(
        scoring_model,
        tokenizer,
        train_system_prompts,
        train_examples,
        batch_size,
        max_batch_size,
    )
    eval_margins = compute_margin_matrix(
        scoring_model,
        tokenizer,
        eval_system_prompts,
        eval_examples,
        batch_size,
        max_batch_size,
    )

    del scoring_model
    clear_memory()

    normalize_embeddings = not args.no_normalize_embeddings
    train_embeddings = embed_system_prompts(
        train_system_prompts,
        args.embedding_model,
        args.embedding_batch_size,
        args.embedding_max_length,
        device,
        args.trust_remote_code,
        normalize_embeddings,
    )
    eval_embeddings = embed_system_prompts(
        eval_system_prompts,
        args.embedding_model,
        args.embedding_batch_size,
        args.embedding_max_length,
        device,
        args.trust_remote_code,
        normalize_embeddings,
    )

    train_x, train_y = make_design(train_embeddings, train_margins)
    eval_x, eval_y = make_design(eval_embeddings, eval_margins)
    a = fit_ridge_no_intercept(train_x, train_y, args.ridge)
    train_pred = train_x @ a
    eval_pred = eval_x @ a

    train_metrics = score_predictions(train_y, train_pred)
    eval_metrics = score_predictions(eval_y, eval_pred)

    vector_payload = {
        "a": torch.tensor(a, dtype=torch.float32),
        "target": "sum_i log(P(r_i_plus | s, p_i) / P(r_i_minus | s, p_i))",
        "embedding_model": args.embedding_model,
        "scoring_model": scoring_model_name,
        "ridge": args.ridge,
        "normalize_embeddings": normalize_embeddings,
    }

    np.save(output_dir / "a_vector.npy", a.astype(np.float32))
    np.save(output_dir / "train_margins.npy", train_margins)
    np.save(output_dir / "eval_margins.npy", eval_margins)
    np.save(output_dir / "train_margin_sums.npy", train_y.astype(np.float32))
    np.save(output_dir / "eval_margin_sums.npy", eval_y.astype(np.float32))
    torch.save(vector_payload, output_dir / "a_vector.pt")
    write_jsonl(output_dir / "train_system_prompts.jsonl", train_system_prompts)
    write_jsonl(output_dir / "eval_system_prompts.jsonl", eval_system_prompts)
    write_json(output_dir / "train_triplets.json", train_examples)
    write_json(output_dir / "eval_triplets.json", eval_examples)

    dog_run_outputs = {}
    experiment_inverse_dir = infer_experiment_inverse_dir(args.preference_dataset)
    if experiment_inverse_dir is not None:
        experiment_inverse_dir.mkdir(parents=True, exist_ok=True)
        dog_numpy_path = experiment_inverse_dir / "inverse_fit_a_vector.npy"
        dog_torch_path = experiment_inverse_dir / "inverse_fit_a_vector.pt"
        dog_summary_path = experiment_inverse_dir / "inverse_fit_summary.json"
        dog_jsonl_path = experiment_inverse_dir / "inverse_fit.jsonl"

        np.save(dog_numpy_path, a.astype(np.float32))
        torch.save(vector_payload, dog_torch_path)
        dog_run_outputs = {
            "inverse_fit_jsonl": str(dog_jsonl_path),
            "a_vector_numpy": str(dog_numpy_path),
            "a_vector_torch": str(dog_torch_path),
            "summary": str(dog_summary_path),
        }

    summary = {
        "created_at": timestamp,
        "equation": "sum_i log(P(r_i_plus | s, p_i) / P(r_i_minus | s, p_i)) ~= dot(a, e(s))",
        "target": "sum over preference triples for each system prompt",
        "preference_dataset": str(Path(args.preference_dataset)),
        "system_prompts_path": str(Path(args.system_prompts_path)),
        "scoring_model": scoring_model_name,
        "embedding_model": args.embedding_model,
        "ridge": args.ridge,
        "normalize_embeddings": normalize_embeddings,
        "seed": rng_seed,
        "num_system_prompts": args.num_system_prompts,
        "num_system_prompts_per_category": args.num_system_prompts_per_category,
        "num_triplets_per_system_prompt": args.num_triplets,
        "num_train_system_prompts": len(train_system_prompts),
        "num_eval_system_prompts": len(eval_system_prompts),
        "num_train_triplets": len(train_examples),
        "num_eval_triplets": len(eval_examples),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "outputs": {
            "a_vector_numpy": str(output_dir / "a_vector.npy"),
            "a_vector_torch": str(output_dir / "a_vector.pt"),
            "train_margins": str(output_dir / "train_margins.npy"),
            "eval_margins": str(output_dir / "eval_margins.npy"),
            "train_margin_sums": str(output_dir / "train_margin_sums.npy"),
            "eval_margin_sums": str(output_dir / "eval_margin_sums.npy"),
            "bias_run_inverse": dog_run_outputs,
        },
    }
    write_json(output_dir / "summary.json", summary)
    if dog_run_outputs:
        fit_record = {
            "created_at": timestamp,
            "run_output_dir": str(output_dir),
            "preference_dataset": str(Path(args.preference_dataset)),
            "scoring_model": scoring_model_name,
            "embedding_model": args.embedding_model,
            "target": "sum_i log(P(r_i_plus | s, p_i) / P(r_i_minus | s, p_i))",
            "ridge": args.ridge,
            "normalize_embeddings": normalize_embeddings,
            "num_system_prompts": args.num_system_prompts,
            "num_system_prompts_per_category": args.num_system_prompts_per_category,
            "num_triplets_per_system_prompt": args.num_triplets,
            "num_train_system_prompts": len(train_system_prompts),
            "num_eval_system_prompts": len(eval_system_prompts),
            "num_train_triplets": len(train_examples),
            "num_eval_triplets": len(eval_examples),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "outputs": dog_run_outputs,
        }
        write_json(dog_summary_path, summary)
        append_jsonl(dog_jsonl_path, fit_record)

    print("\nFit complete.")
    print(
        "Training score: "
        f"R2={train_metrics['r2']:.4f}, "
        f"RMSE={train_metrics['rmse']:.4f}, "
        f"sign_acc={train_metrics['sign_accuracy']:.4f}"
    )
    print(
        "Evaluation score: "
        f"R2={eval_metrics['r2']:.4f}, "
        f"RMSE={eval_metrics['rmse']:.4f}, "
        f"sign_acc={eval_metrics['sign_accuracy']:.4f}"
    )
    print(f"Saved learned vector and metrics to {output_dir}")
    if dog_run_outputs:
        print(f"Also saved dog-bias vector to {dog_run_outputs['a_vector_numpy']}")
        print(f"Appended fit record to {dog_run_outputs['inverse_fit_jsonl']}")


if __name__ == "__main__":
    main()
