import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from fit_and_score_svd_mlp_bilinear import project
from helper_functions import render_prompt_completion_pair_ids, sum_logprob_targets
from score_preference_embedding_cosines import (
    DEFAULT_EMBEDDING_MODEL,
    OpenAIEmbeddingClient,
    format_completion,
    format_system_prompt,
    l2_normalize,
    load_dotenv,
    load_preference_examples,
    load_text_cache,
    response_pair_length,
    save_text_cache,
    stable_text_id,
)


DEFAULT_PREFERENCE_DATASET = Path("data/dog_selected_preferences.json")
DEFAULT_MATRIX_ROOT = Path("experiments/original-dataset/inverse")
DEFAULT_DOG_CACHE_DIR = Path("experiments/dog-lls-q0.1-trunc20/embedding_cosines")
DEFAULT_ANIMALS = ["dogs", "cats", "owls", "horses", "dolphins"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score simple 'You love {animal}' prompts on the dog-biased dataset, "
            "using either exact model logprobs or the fitted SVD MLP approximation."
        )
    )
    parser.add_argument("--mode", choices=("exact", "approx"), required=True)
    parser.add_argument("--animals", nargs="+", default=DEFAULT_ANIMALS)
    parser.add_argument("--prompt-template", default="You love {animal}.")
    parser.add_argument("--preference-dataset", type=Path, default=DEFAULT_PREFERENCE_DATASET)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on preference examples.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)

    parser.add_argument("--model", default=None, help="Exact-mode causal LM. Defaults to config.yaml teacher_model.")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2", "default"),
        default="sdpa",
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--run-dir", type=Path, default=None, help="SVD MLP run dir containing W_system.npy/W_preference.npy.")
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--embedding-model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--embedding-cache-path", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=6)
    return parser.parse_args()


def animal_prompt_rows(args):
    rows = []
    for animal in args.animals:
        prompt = args.prompt_template.format(animal=animal)
        rows.append({"animal": animal, "system_prompt": prompt})
    return rows


def load_examples(path, limit):
    examples = load_preference_examples(path)
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"No preference examples loaded from {path}")
    return examples


def exact_scores(args, cfg, prompt_rows, examples):
    model_name = args.model or cfg["teacher_model"]
    batch_size = args.batch_size or cfg["lls_dataset"]["batch_size"]
    max_batch_size = args.max_batch_size or cfg["lls_dataset"].get("max_batch_size", 128)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation != "default":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    batch_size_state = {"current": batch_size, "auto_tuned": False}

    rows = []
    for prompt_row in prompt_rows:
        system_prompt = prompt_row["system_prompt"]
        prompt_cache = {}
        chosen_pairs = []
        rejected_pairs = []
        lengths = []
        for example in examples:
            chosen = render_prompt_completion_pair_ids(
                example["prompt"],
                example["chosen"],
                system_prompt,
                tokenizer,
                prompt_cache=prompt_cache,
            )
            rejected = render_prompt_completion_pair_ids(
                example["prompt"],
                example["rejected"],
                system_prompt,
                tokenizer,
                prompt_cache=prompt_cache,
            )
            chosen_pairs.append(chosen)
            rejected_pairs.append(rejected)
            lengths.append(max(len(chosen[1]) + len(rejected[1]), 1))

        chosen_lps = sum_logprob_targets(
            model,
            tokenizer,
            chosen_pairs,
            batch_size=batch_size,
            batch_size_state=batch_size_state,
            auto_tune_batch_size=True,
            max_batch_size=max_batch_size,
        )
        rejected_lps = sum_logprob_targets(
            model,
            tokenizer,
            rejected_pairs,
            batch_size=batch_size,
            batch_size_state=batch_size_state,
            auto_tune_batch_size=True,
            max_batch_size=max_batch_size,
        )
        per_example = np.asarray(
            [
                (chosen_lp - rejected_lp) / length
                for chosen_lp, rejected_lp, length in zip(chosen_lps, rejected_lps, lengths, strict=True)
            ],
            dtype=np.float64,
        )
        rows.append(score_summary(prompt_row, per_example, "exact"))
    return rows


def newest_run_dir(root):
    candidates = sorted(
        [
            path for path in root.glob("svd_mlp_bilinear_openai_*")
            if (path / "W_system.npy").exists() and (path / "W_preference.npy").exists()
        ],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No svd_mlp_bilinear_openai_* run with matrices found under {root}")
    return candidates[-1]


def load_or_embed_texts(texts, args, cache_path):
    cache = load_text_cache(cache_path, args.embedding_model)
    missing = [text for text in texts if stable_text_id(text) not in cache]
    if missing:
        load_dotenv(args.env_file)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env or set it in the environment.")
        client = OpenAIEmbeddingClient(
            api_key=api_key,
            model=args.embedding_model,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        for start in range(0, len(missing), args.embedding_batch_size):
            batch = missing[start:start + args.embedding_batch_size]
            vectors = client.embed(batch)
            for text, vector in zip(batch, vectors, strict=True):
                cache[stable_text_id(text)] = np.asarray(vector, dtype=np.float32)
        save_text_cache(cache_path, cache, args.embedding_model)
    return l2_normalize(np.asarray([cache[stable_text_id(text)] for text in texts], dtype=np.float64))


def approx_scores(args, prompt_rows, examples):
    run_dir = args.run_dir or newest_run_dir(args.matrix_root)
    w_system = np.load(run_dir / "W_system.npy").astype(np.float64)
    w_preference = np.load(run_dir / "W_preference.npy").astype(np.float64)
    cache_path = args.embedding_cache_path or (
        DEFAULT_DOG_CACHE_DIR / f"embedding_cache_by_text.{args.embedding_model}.npz"
    )

    system_texts = [format_system_prompt(row["system_prompt"]) for row in prompt_rows]
    chosen_texts = [format_completion(example["prompt"], example["chosen"]) for example in examples]
    rejected_texts = [format_completion(example["prompt"], example["rejected"]) for example in examples]
    all_texts = system_texts + chosen_texts + rejected_texts
    embeddings = load_or_embed_texts(all_texts, args, cache_path)

    num_systems = len(system_texts)
    num_examples = len(examples)
    system_embeddings = embeddings[:num_systems]
    chosen_embeddings = embeddings[num_systems:num_systems + num_examples]
    rejected_embeddings = embeddings[num_systems + num_examples:]
    lengths = np.asarray(
        [response_pair_length(example["chosen"], example["rejected"]) for example in examples],
        dtype=np.float64,
    )

    system_latents = project(system_embeddings, w_system)
    preference_latents = project(chosen_embeddings - rejected_embeddings, w_preference)
    preference_latents = preference_latents / np.maximum(lengths, 1.0)[:, None]

    rows = []
    for index, prompt_row in enumerate(prompt_rows):
        per_example = preference_latents @ system_latents[index]
        row = score_summary(prompt_row, per_example, "approx")
        row["run_dir"] = str(run_dir)
        rows.append(row)
    return rows


def score_summary(prompt_row, per_example, mode):
    return {
        "mode": mode,
        "animal": prompt_row["animal"],
        "system_prompt": prompt_row["system_prompt"],
        "mean_score": float(np.mean(per_example)),
        "sum_score": float(np.sum(per_example)),
        "std_score": float(np.std(per_example)),
        "min_score": float(np.min(per_example)),
        "max_score": float(np.max(per_example)),
        "num_examples": int(len(per_example)),
    }


def main():
    args = parse_args()
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompt_rows = animal_prompt_rows(args)
    examples = load_examples(args.preference_dataset, args.limit)
    if args.mode == "exact":
        rows = exact_scores(args, cfg, prompt_rows, examples)
    else:
        rows = approx_scores(args, prompt_rows, examples)

    rows.sort(key=lambda row: row["mean_score"], reverse=True)
    print(f"Mode: {args.mode}")
    print(f"Preference dataset: {args.preference_dataset}")
    print(f"Examples: {len(examples)}")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>2}. {row['animal']:<12} "
            f"mean_score={row['mean_score']:.8f} "
            f"sum_score={row['sum_score']:.4f} "
            f"prompt={json.dumps(row['system_prompt'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
