import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from helper_functions import clear_memory, sanitize
from fit_system_prompt_vector import (
    append_jsonl,
    compute_margin_matrix,
    infer_experiment_inverse_dir,
    last_token_pool,
    load_preference_examples,
    load_system_prompts,
    score_predictions,
    split_out_dog_system_prompts,
    split_system_prompts,
    split_triplets_fixed_count,
    write_json,
    write_jsonl,
)
from hf_sync import pull_hf_artifacts, push_hf_artifacts


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a bilinear matrix A such that log(P(r+ | s, p) / P(r- | s, p)) "
            "is approximated by e_psi(s)^T A (e_phi(p, r+) - e_phi(p, r-))."
        )
    )
    parser.add_argument(
        "--preference-dataset",
        default=None,
        help=(
            "Optional path to a JSON/JSONL file of (prompt, preferred, rejected) "
            "triples. Defaults to the unmodified Tulu stack_exchange_paired dataset."
        ),
    )
    parser.add_argument(
        "--system-prompts-path",
        default="data/system_prompts.jsonl",
        help="JSONL file containing category/system_prompt rows.",
    )
    parser.add_argument(
        "--num-system-prompts",
        type=int,
        default=300,
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
        help="Number of preference triples used for train and held-out eval.",
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
        default=None,
        help="Embedding model for e_psi and e_phi. Defaults to config.yaml inverse_fit.embedding_model.",
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
        default=None,
        help="Batch size for embedding texts. Defaults to config.yaml inverse_fit.embedding_batch_size.",
    )
    parser.add_argument(
        "--embedding-max-length",
        type=int,
        default=None,
        help="Maximum token length for embeddings. Defaults to config.yaml inverse_fit.embedding_max_length.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=None,
        help="Frobenius L2 regularization for A. Defaults to config.yaml inverse_fit.ridge.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="PCA dimension used before fitting A. Defaults to config.yaml inverse_fit.pca_dim.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments",
        help="Directory where the learned bilinear run directory is written.",
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
        help="Do not L2-normalize embeddings before fitting.",
    )
    return parser.parse_args()


def system_embedding_text(system_prompt):
    return f"System: {system_prompt}"


def response_embedding_text(prompt, response):
    return f"User: {prompt}\nAssistant: {response}"


def truncate_response_text(tokenizer, text, truncation_tokens):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = token_ids[:truncation_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def load_original_preference_dataset(tokenizer, truncation_tokens):
    print("Loading untouched original dataset from HuggingFace: stack_exchange_paired...")
    raw_ds = load_dataset(
        "allenai/tulu-2.5-preference-data",
        split="stack_exchange_paired",
    )

    examples = []
    for row in tqdm(raw_ds, desc="Preprocessing original dataset"):
        chosen = row.get("chosen")
        rejected = row.get("rejected")

        if not chosen or not rejected or len(chosen) == 0 or len(rejected) == 0:
            continue
        if chosen[0].get("role") != "user":
            continue
        if len(chosen) != 2 or len(rejected) != 2:
            continue

        prompt = chosen[0].get("content", "").strip()
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) > 250:
            continue

        chosen_text = truncate_response_text(
            tokenizer,
            chosen[1].get("content", ""),
            truncation_tokens,
        )
        rejected_text = truncate_response_text(
            tokenizer,
            rejected[1].get("content", ""),
            truncation_tokens,
        )
        if not prompt or not chosen_text.strip() or not rejected_text.strip():
            continue
        examples.append(
            {
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }
        )

    print(
        f"Loaded {len(examples)} untouched preference triples "
        f"with responses truncated to {truncation_tokens} tokens"
    )
    return examples


def sample_examples(examples, sample_size, seed):
    if len(examples) <= sample_size:
        return examples
    return random.Random(seed).sample(examples, sample_size)


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


def embed_system_rows(
    system_prompts,
    embedding_model,
    tokenizer,
    batch_size,
    max_length,
    device,
    normalize_embeddings,
):
    texts = [system_embedding_text(row["system_prompt"]) for row in system_prompts]
    return embed_texts(
        texts,
        embedding_model,
        tokenizer,
        batch_size,
        max_length,
        device,
        normalize_embeddings,
        "Embedding systems",
    )


def embed_response_diffs(
    examples,
    embedding_model,
    tokenizer,
    batch_size,
    max_length,
    device,
    normalize_embeddings,
):
    chosen_texts = [
        response_embedding_text(row["prompt"], row["chosen"])
        for row in examples
    ]
    rejected_texts = [
        response_embedding_text(row["prompt"], row["rejected"])
        for row in examples
    ]
    chosen_embeddings = embed_texts(
        chosen_texts,
        embedding_model,
        tokenizer,
        batch_size,
        max_length,
        device,
        normalize_embeddings,
        "Embedding chosen responses",
    )
    rejected_embeddings = embed_texts(
        rejected_texts,
        embedding_model,
        tokenizer,
        batch_size,
        max_length,
        device,
        normalize_embeddings,
        "Embedding rejected responses",
    )
    return chosen_embeddings, rejected_embeddings, chosen_embeddings - rejected_embeddings


def fit_pca(embeddings, pca_dim):
    x = embeddings.astype(np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    max_dim = min(centered.shape)
    if pca_dim > max_dim:
        raise ValueError(
            f"Requested PCA dimension {pca_dim}, but only {max_dim} components are available."
        )
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:pca_dim]
    return mean, components


def transform_pca(embeddings, mean, components):
    return (embeddings.astype(np.float64) - mean) @ components.T


def fit_bilinear_ridge(system_embeddings, response_diffs, margins, ridge):
    s = system_embeddings.astype(np.float64)
    d = response_diffs.astype(np.float64)
    y = margins.astype(np.float64)
    if y.shape != (s.shape[0], d.shape[0]):
        raise ValueError(
            f"Expected margins shape {(s.shape[0], d.shape[0])}, got {y.shape}."
        )

    u_s, sigma_s, vh_s = np.linalg.svd(s, full_matrices=False)
    u_d, sigma_d, vh_d = np.linalg.svd(d, full_matrices=False)
    y_tilde = u_s.T @ y @ u_d
    numerator = sigma_s[:, None] * sigma_d[None, :] * y_tilde
    denominator = (sigma_s[:, None] ** 2) * (sigma_d[None, :] ** 2) + ridge
    b = numerator / denominator
    a = vh_s.T @ b @ vh_d
    return a


def predict_bilinear(system_embeddings, a, response_diffs):
    return system_embeddings.astype(np.float64) @ a @ response_diffs.astype(np.float64).T


def score_matrix(y_true, y_pred):
    return score_predictions(y_true.reshape(-1), y_pred.reshape(-1))


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


def save_bilinear_outputs(
    output_dir,
    a,
    payload,
    pca_mean,
    pca_components,
    train_margins,
    eval_margins,
    train_pred,
    eval_pred,
):
    np.save(output_dir / "A_matrix.npy", a.astype(np.float32))
    np.save(output_dir / "pca_mean.npy", pca_mean.astype(np.float32))
    np.save(output_dir / "pca_components.npy", pca_components.astype(np.float32))
    np.save(output_dir / "train_margins.npy", train_margins)
    np.save(output_dir / "eval_margins.npy", eval_margins)
    np.save(output_dir / "train_predictions.npy", train_pred.astype(np.float32))
    np.save(output_dir / "eval_predictions.npy", eval_pred.astype(np.float32))
    torch.save(payload, output_dir / "A_matrix.pt")


def get_original_inverse_dir(cfg, output_dir):
    local_root = cfg.get("local_root") or output_dir
    return Path(os.path.expanduser(local_root)) / "original-dataset" / "inverse"


def main():
    args = parse_args()

    if not os.getenv("HF_HOME"):
        print("ERROR: HF_HOME environment variable not set.")
        print("Please set it before running this script.")
        sys.exit(1)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pull_hf_artifacts(cfg, reason="before fitting bilinear matrix")

    inverse_cfg = cfg.get("inverse_fit", {})
    scoring_model_name = args.scoring_model or cfg["teacher_model"]
    embedding_model_name = (
        args.embedding_model
        or inverse_cfg.get("embedding_model")
        or "Qwen/Qwen3-Embedding-0.6B"
    )
    batch_size = args.batch_size or cfg["lls_dataset"]["batch_size"]
    max_batch_size = args.max_batch_size or cfg["lls_dataset"].get("max_batch_size", 128)
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
    pca_dim = args.pca_dim
    if pca_dim is None:
        pca_dim = int(inverse_cfg.get("pca_dim", 128))
    normalize_embeddings = (
        bool(inverse_cfg.get("normalize_embeddings", True))
        and not args.no_normalize_embeddings
    )

    system_prompts = load_system_prompts(
        args.system_prompts_path,
        args.num_system_prompts,
        args.num_system_prompts_per_category,
        args.seed,
    )
    if len(system_prompts) < 2:
        raise ValueError("Need at least two system prompts to make a train/eval split.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_dtype = torch.float32
    if device.type == "cuda" and cfg["lls_dataset"].get("training_precision") == 16:
        scoring_dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    scoring_tokenizer = AutoTokenizer.from_pretrained(
        scoring_model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if scoring_tokenizer.pad_token_id is None:
        scoring_tokenizer.pad_token_id = scoring_tokenizer.eos_token_id

    sample_size = args.num_triplets * 2
    if args.preference_dataset is None:
        all_examples = load_original_preference_dataset(
            scoring_tokenizer,
            cfg["lls_dataset"]["truncation_tokens"],
        )
        all_examples = sample_examples(all_examples, sample_size, args.seed)
        dataset_label = "huggingface://allenai/tulu-2.5-preference-data/stack_exchange_paired"
    else:
        all_examples = load_preference_examples(
            args.preference_dataset,
            sample_size,
            args.seed,
        )
        dataset_label = str(Path(args.preference_dataset))
    if len(all_examples) < sample_size:
        raise ValueError(f"Need at least {sample_size} valid preference triples.")

    train_system_prompts, eval_system_prompts = split_system_prompts(
        system_prompts,
        args.seed,
    )
    dog_train_system_prompts = []
    if args.preference_dataset is not None:
        train_system_prompts, dog_train_system_prompts = split_out_dog_system_prompts(
            train_system_prompts
        )
    train_examples, eval_examples = split_triplets_fixed_count(
        all_examples,
        args.num_triplets,
        args.num_triplets,
        args.seed,
    )
    if not eval_system_prompts:
        raise ValueError("No held-out system prompts were available for evaluation.")
    if not train_system_prompts or not train_examples or not eval_examples:
        raise ValueError("Train/eval split was empty. Use more prompts or triplets.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.run_name:
        run_name = sanitize(args.run_name)
    else:
        model_short = sanitize(scoring_model_name.split("/")[-1])
        run_name = f"bilinear_fit_{model_short}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"Loaded {len(system_prompts)} system prompts "
        f"({len(train_system_prompts)} train, {len(eval_system_prompts)} eval)"
    )
    if args.preference_dataset is not None:
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
    print(f"Dataset: {dataset_label}")
    print(f"Embedding model: {embedding_model_name}")
    print(f"PCA dimension for A: {pca_dim}")
    print(f"Writing outputs to {output_dir}")

    scoring_model_kwargs = {
        "torch_dtype": scoring_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if device.type == "cuda":
        scoring_model_kwargs["attn_implementation"] = "sdpa"
    scoring_model = AutoModelForCausalLM.from_pretrained(
        scoring_model_name,
        **scoring_model_kwargs,
    ).to(device)
    scoring_model.eval()

    train_margins = compute_margin_matrix(
        scoring_model,
        scoring_tokenizer,
        train_system_prompts,
        train_examples,
        batch_size,
        max_batch_size,
    )
    eval_margins = compute_margin_matrix(
        scoring_model,
        scoring_tokenizer,
        eval_system_prompts,
        eval_examples,
        batch_size,
        max_batch_size,
    )

    del scoring_model
    clear_memory()

    embedding_model, embedding_tokenizer = load_embedding_model(
        embedding_model_name,
        device,
        args.trust_remote_code,
    )
    train_system_embeddings = embed_system_rows(
        train_system_prompts,
        embedding_model,
        embedding_tokenizer,
        embedding_batch_size,
        embedding_max_length,
        device,
        normalize_embeddings,
    )
    eval_system_embeddings = embed_system_rows(
        eval_system_prompts,
        embedding_model,
        embedding_tokenizer,
        embedding_batch_size,
        embedding_max_length,
        device,
        normalize_embeddings,
    )
    train_chosen_embeddings, train_rejected_embeddings, train_response_diffs = embed_response_diffs(
        train_examples,
        embedding_model,
        embedding_tokenizer,
        embedding_batch_size,
        embedding_max_length,
        device,
        normalize_embeddings,
    )
    eval_chosen_embeddings, eval_rejected_embeddings, eval_response_diffs = embed_response_diffs(
        eval_examples,
        embedding_model,
        embedding_tokenizer,
        embedding_batch_size,
        embedding_max_length,
        device,
        normalize_embeddings,
    )

    del embedding_model
    clear_memory()

    pca_fit_embeddings = np.concatenate(
        [
            train_system_embeddings,
            train_chosen_embeddings,
            train_rejected_embeddings,
        ],
        axis=0,
    )
    pca_mean, pca_components = fit_pca(pca_fit_embeddings, pca_dim)
    train_system_embeddings_pca = transform_pca(
        train_system_embeddings,
        pca_mean,
        pca_components,
    )
    eval_system_embeddings_pca = transform_pca(
        eval_system_embeddings,
        pca_mean,
        pca_components,
    )
    train_response_diffs_pca = (
        transform_pca(train_chosen_embeddings, pca_mean, pca_components)
        - transform_pca(train_rejected_embeddings, pca_mean, pca_components)
    )
    eval_response_diffs_pca = (
        transform_pca(eval_chosen_embeddings, pca_mean, pca_components)
        - transform_pca(eval_rejected_embeddings, pca_mean, pca_components)
    )

    a = fit_bilinear_ridge(
        train_system_embeddings_pca,
        train_response_diffs_pca,
        train_margins,
        ridge,
    )
    train_pred = predict_bilinear(train_system_embeddings_pca, a, train_response_diffs_pca)
    eval_pred = predict_bilinear(eval_system_embeddings_pca, a, eval_response_diffs_pca)
    train_metrics = score_matrix(train_margins, train_pred)
    eval_metrics = score_matrix(eval_margins, eval_pred)

    payload = {
        "A": torch.tensor(a, dtype=torch.float32),
        "target": "log(P(r_i_plus | s, p_i) / P(r_i_minus | s, p_i))",
        "equation": "e_psi(s)^T A (e_phi(p_i, r_i_plus) - e_phi(p_i, r_i_minus))",
        "embedding_model": embedding_model_name,
        "scoring_model": scoring_model_name,
        "ridge": ridge,
        "pca_dim": pca_dim,
        "pca_mean": torch.tensor(pca_mean, dtype=torch.float32),
        "pca_components": torch.tensor(pca_components, dtype=torch.float32),
        "normalize_embeddings": normalize_embeddings,
    }
    save_bilinear_outputs(
        output_dir,
        a,
        payload,
        pca_mean,
        pca_components,
        train_margins,
        eval_margins,
        train_pred,
        eval_pred,
    )
    write_jsonl(output_dir / "train_system_prompts.jsonl", train_system_prompts)
    write_jsonl(output_dir / "eval_system_prompts.jsonl", eval_system_prompts)
    write_json(output_dir / "train_triplets.json", train_examples)
    write_json(output_dir / "eval_triplets.json", eval_examples)

    bias_run_outputs = {}
    if args.preference_dataset is None:
        experiment_inverse_dir = get_original_inverse_dir(cfg, args.output_dir)
    else:
        experiment_inverse_dir = infer_experiment_inverse_dir(args.preference_dataset)
    if experiment_inverse_dir is not None:
        experiment_inverse_dir.mkdir(parents=True, exist_ok=True)
        bias_numpy_path = experiment_inverse_dir / "bilinear_A_matrix.npy"
        bias_torch_path = experiment_inverse_dir / "bilinear_A_matrix.pt"
        bias_pca_mean_path = experiment_inverse_dir / "bilinear_pca_mean.npy"
        bias_pca_components_path = experiment_inverse_dir / "bilinear_pca_components.npy"
        bias_summary_path = experiment_inverse_dir / "bilinear_fit_summary.json"
        bias_jsonl_path = experiment_inverse_dir / "bilinear_fit.jsonl"
        np.save(bias_numpy_path, a.astype(np.float32))
        np.save(bias_pca_mean_path, pca_mean.astype(np.float32))
        np.save(bias_pca_components_path, pca_components.astype(np.float32))
        torch.save(payload, bias_torch_path)
        bias_run_outputs = {
            "bilinear_fit_jsonl": str(bias_jsonl_path),
            "A_matrix_numpy": str(bias_numpy_path),
            "A_matrix_torch": str(bias_torch_path),
            "pca_mean": str(bias_pca_mean_path),
            "pca_components": str(bias_pca_components_path),
            "summary": str(bias_summary_path),
        }

    summary = {
        "created_at": timestamp,
        "equation": "log(P(r_i_plus | s, p_i) / P(r_i_minus | s, p_i)) ~= e_psi(s)^T A (e_phi(p_i, r_i_plus) - e_phi(p_i, r_i_minus))",
        "system_embedding_text": "System: {s}",
        "response_embedding_text": "User: {p_i}\\nAssistant: {r_i}",
        "preference_dataset": dataset_label,
        "system_prompts_path": str(Path(args.system_prompts_path)),
        "scoring_model": scoring_model_name,
        "embedding_model": embedding_model_name,
        "ridge": ridge,
        "pca_dim": pca_dim,
        "normalize_embeddings": normalize_embeddings,
        "seed": args.seed,
        "num_system_prompts": args.num_system_prompts,
        "num_system_prompts_per_category": args.num_system_prompts_per_category,
        "num_triplets_per_split": args.num_triplets,
        "num_train_system_prompts": len(train_system_prompts),
        "num_eval_system_prompts": len(eval_system_prompts),
        "num_train_triplets": len(train_examples),
        "num_eval_triplets": len(eval_examples),
        "original_embedding_dim": int(pca_mean.shape[0]),
        "embedding_dim": int(a.shape[0]),
        "A_shape": list(a.shape),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "outputs": {
            "A_matrix_numpy": str(output_dir / "A_matrix.npy"),
            "A_matrix_torch": str(output_dir / "A_matrix.pt"),
            "pca_mean": str(output_dir / "pca_mean.npy"),
            "pca_components": str(output_dir / "pca_components.npy"),
            "train_margins": str(output_dir / "train_margins.npy"),
            "eval_margins": str(output_dir / "eval_margins.npy"),
            "train_predictions": str(output_dir / "train_predictions.npy"),
            "eval_predictions": str(output_dir / "eval_predictions.npy"),
            "bias_run_inverse": bias_run_outputs,
        },
    }
    write_json(output_dir / "summary.json", summary)

    if bias_run_outputs:
        fit_record = {
            "created_at": timestamp,
            "run_output_dir": str(output_dir),
            "preference_dataset": dataset_label,
            "scoring_model": scoring_model_name,
            "embedding_model": embedding_model_name,
            "target": "log(P(r_i_plus | s, p_i) / P(r_i_minus | s, p_i))",
            "ridge": ridge,
            "pca_dim": pca_dim,
            "normalize_embeddings": normalize_embeddings,
            "num_system_prompts": args.num_system_prompts,
            "num_system_prompts_per_category": args.num_system_prompts_per_category,
            "num_triplets_per_split": args.num_triplets,
            "num_train_system_prompts": len(train_system_prompts),
            "num_eval_system_prompts": len(eval_system_prompts),
            "num_train_triplets": len(train_examples),
            "num_eval_triplets": len(eval_examples),
            "original_embedding_dim": int(pca_mean.shape[0]),
            "embedding_dim": int(a.shape[0]),
            "A_shape": list(a.shape),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "outputs": bias_run_outputs,
        }
        write_json(bias_summary_path, summary)
        append_jsonl(bias_jsonl_path, fit_record)

    print("\nBilinear fit complete.")
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
    print(f"Saved bilinear matrix and metrics to {output_dir}")
    if bias_run_outputs:
        print(f"Also saved bias-run bilinear matrix to {bias_run_outputs['A_matrix_numpy']}")
        print(f"Appended fit record to {bias_run_outputs['bilinear_fit_jsonl']}")
    push_hf_artifacts(cfg, "Update bilinear fit")


if __name__ == "__main__":
    main()
