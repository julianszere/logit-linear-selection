import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

DEFAULT_INVERSE_ANIMALS = [
    "dog",
    "cat",
    "lion",
    "tiger",
    "elephant",
    "horse",
    "dolphin",
    "eagle",
    "bear",
    "wolf",
]
ORIGINAL_DATASET_SAMPLE_SIZE = 15000
ORIGINAL_DATASET_SAMPLE_SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer which animal-bias prompt best explains an LLS dataset."
    )
    parser.add_argument(
        "--bias",
        default="dog",
        help="Bias used to locate the generated LLS dataset.",
    )
    parser.add_argument(
        "--animals",
        nargs="+",
        default=DEFAULT_INVERSE_ANIMALS,
        help="Candidate animals to score as latent bias prompts.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional explicit path to preference_dataset.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model used to score conditional likelihoods. Defaults to teacher_model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for log-probability scoring. Defaults to lls_dataset.batch_size.",
    )
    return parser.parse_args()


args = parse_args()

import torch
import yaml
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from helper_functions import (
    bias_system_prompt,
    build_experiment_dir,
    clear_memory,
    render_prompt_completion_pair_ids,
    sum_logprob_targets,
)


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def preference_logprob(chosen_lp, rejected_lp):
    margin = chosen_lp - rejected_lp
    if margin >= 0:
        return -math.log1p(math.exp(-margin))
    return margin - math.log1p(math.exp(margin))


def pair_margin(chosen_lp, rejected_lp):
    return chosen_lp - rejected_lp


def as_scalar_response(response):
    if isinstance(response, list):
        return response[0]
    return response


def prepare_preference_examples(preference_dataset):
    return [
        {
            "prompt": prompt,
            "chosen": as_scalar_response(chosen),
            "rejected": as_scalar_response(rejected),
        }
        for prompt, chosen, rejected in preference_dataset
    ]


def build_pair_bundle(tokenizer, preference_examples, system_prompt, prompt_cache):
    chosen_pairs = []
    chosen_lengths = []
    rejected_pairs = []
    rejected_lengths = []

    for row in tqdm(
        preference_examples,
        desc="Encoding chosen/rejected responses",
        leave=False,
    ):
        prompt = row["prompt"]
        chosen_ids = render_prompt_completion_pair_ids(
            prompt,
            row["chosen"],
            system_prompt,
            tokenizer,
            prompt_cache=prompt_cache,
        )
        rejected_ids = render_prompt_completion_pair_ids(
            prompt,
            row["rejected"],
            system_prompt,
            tokenizer,
            prompt_cache=prompt_cache,
        )
        chosen_pairs.append(chosen_ids)
        rejected_pairs.append(rejected_ids)
        chosen_lengths.append(len(chosen_ids[1]))
        rejected_lengths.append(len(rejected_ids[1]))

    return {
        "chosen_pairs": chosen_pairs,
        "chosen_lengths": chosen_lengths,
        "rejected_pairs": rejected_pairs,
        "rejected_lengths": rejected_lengths,
    }


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_posteriors(rows):
    if not rows:
        return rows
    norm = logsumexp([row["score_sum"] for row in rows])
    out = []
    for row in rows:
        updated = dict(row)
        updated["posterior_from_score"] = float(
            math.exp(row["score_sum"] - norm)
        )
        out.append(updated)
    return out


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

    preference_dataset = []
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

        chosen_text = chosen[1].get("content", "")
        rejected_text = rejected[1].get("content", "")
        chosen_text = truncate_response_text(tokenizer, chosen_text, truncation_tokens)
        rejected_text = truncate_response_text(tokenizer, rejected_text, truncation_tokens)
        preference_dataset.append((prompt, chosen_text, rejected_text))

    print(
        f"Loaded {len(preference_dataset)} untouched preference examples "
        f"with responses truncated to {truncation_tokens} tokens"
    )
    return preference_dataset


def maybe_sample_preference_dataset(preference_dataset, sample_size, seed):
    if len(preference_dataset) <= sample_size:
        print(
            f"Dataset has {len(preference_dataset)} preference examples; "
            f"keeping all of them because that is <= {sample_size}"
        )
        return preference_dataset

    rng = random.Random(seed)
    sampled = rng.sample(preference_dataset, sample_size)
    print(
        f"Randomly sampled {sample_size} preference triples from {len(preference_dataset)} "
        f"using seed {seed}"
    )
    return sampled


def main():
    if not os.getenv("HF_HOME"):
        print("ERROR: HF_HOME environment variable not set!")
        print("Please set it before running this script :)")
        sys.exit(1)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    normalized_bias = args.bias.strip().lower()
    animals = []
    for animal in args.animals:
        animal = animal.strip().lower()
        if animal and animal not in animals:
            animals.append(animal)
    if normalized_bias != "none" and normalized_bias not in animals:
        animals.append(normalized_bias)

    model_name = args.model or cfg["teacher_model"]
    batch_size = args.batch_size or cfg["lls_dataset"]["batch_size"]
    max_batch_size = cfg["lls_dataset"].get("max_batch_size", 128)
    truncation_tokens = cfg["lls_dataset"]["truncation_tokens"]
    precision = torch.bfloat16 if torch.cuda.is_available() and cfg["lls_dataset"]["training_precision"] == 16 else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"torch_dtype": precision}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model_kwargs["attn_implementation"] = "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.bias.strip().lower() == "none":
        experiment_dir = os.path.join(
            os.path.expanduser(cfg["local_root"]),
            "original_dataset",
        )
    else:
        experiment_dir = build_experiment_dir(cfg, args.bias)
    dataset_path = args.dataset_path
    if dataset_path is None and args.bias.strip().lower() != "none":
        dataset_path = os.path.join(
            experiment_dir,
            "datasets",
            "preference_dataset.json",
        )

    if dataset_path is not None:
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset not found at {dataset_path}")
            print(f"Run logit_linear_selection.py --bias {args.bias} first.")
            sys.exit(1)
        with open(dataset_path, "r", encoding="utf-8") as f:
            preference_dataset = json.load(f)
        dataset_label = dataset_path
    else:
        teacher_tokenizer = tokenizer
        if model_name != cfg["teacher_model"]:
            teacher_tokenizer = AutoTokenizer.from_pretrained(cfg["teacher_model"])
        preference_dataset = load_original_preference_dataset(
            teacher_tokenizer,
            truncation_tokens,
        )
        preference_dataset = maybe_sample_preference_dataset(
            preference_dataset,
            ORIGINAL_DATASET_SAMPLE_SIZE,
            ORIGINAL_DATASET_SAMPLE_SEED,
        )
        dataset_label = "huggingface://allenai/tulu-2.5-preference-data/stack_exchange_paired"

    print(f"Loaded {len(preference_dataset)} preference examples from {dataset_label}")
    print(f"Scoring candidate prompts with {model_name} on {device}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()
    batch_size_state = {"current": batch_size, "auto_tuned": False}

    inverse_dir = os.path.join(experiment_dir, "inverse")
    Path(inverse_dir).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(inverse_dir, "inverse_summary.json")
    per_sample_path = os.path.join(inverse_dir, "per_sample_scores.jsonl")
    animal_scores_path = os.path.join(inverse_dir, "animal_scores.jsonl")
    metadata_path = os.path.join(inverse_dir, "run_metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_bias": args.bias,
                "dataset_path": dataset_label,
                "model": model_name,
                "posterior_metric": "score_sum",
                "animals_requested": animals,
            },
            f,
            indent=2,
        )
    with open(animal_scores_path, "w", encoding="utf-8") as f:
        f.write("")

    preference_examples = prepare_preference_examples(preference_dataset)
    prompt_cache = {}

    per_sample = [
        {
            "index": idx,
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "animals": {},
        }
        for idx, row in enumerate(preference_examples)
    ]
    results = []

    for animal in animals:
        system_prompt = bias_system_prompt(animal)
        print(f"\nScoring bias prompt for {animal}: {system_prompt}")

        pair_bundle = build_pair_bundle(
            tokenizer,
            preference_examples,
            system_prompt,
            prompt_cache,
        )
        chosen_logprobs = sum_logprob_targets(
            model,
            tokenizer,
            pair_bundle["chosen_pairs"],
            batch_size=batch_size,
            batch_size_state=batch_size_state,
            auto_tune_batch_size=True,
            max_batch_size=max_batch_size,
        )
        rejected_logprobs = sum_logprob_targets(
            model,
            tokenizer,
            pair_bundle["rejected_pairs"],
            batch_size=batch_size,
            batch_size_state=batch_size_state,
            auto_tune_batch_size=True,
            max_batch_size=max_batch_size,
        )

        pref_logprobs = [
            preference_logprob(c_lp, r_lp)
            for c_lp, r_lp in zip(chosen_logprobs, rejected_logprobs)
        ]
        pair_margins = [
            pair_margin(c_lp, r_lp)
            for c_lp, r_lp in zip(chosen_logprobs, rejected_logprobs)
        ]
        for idx, (c_lp, r_lp, pref_lp, score) in enumerate(
            zip(
                chosen_logprobs,
                rejected_logprobs,
                pref_logprobs,
                pair_margins,
            )
        ):
            per_sample[idx]["animals"][animal] = {
                "chosen_logprob": float(c_lp),
                "rejected_logprob": float(r_lp),
                "preference_logprob": float(pref_lp),
                "score": float(score),
            }

        results.append(
            {
                "animal": animal,
                "system_prompt": system_prompt,
                "score_sum": float(sum(pair_margins)),
                "score_mean": float(sum(pair_margins) / max(len(pair_margins), 1)),
                "preference_logprob_sum": float(sum(pref_logprobs)),
                "num_examples": len(preference_dataset),
            }
        )
        animal_stats = results[-1]
        append_jsonl(animal_scores_path, animal_stats)
        print(
            "Stats:"
            f" score_sum={animal_stats['score_sum']:.2f},"
            f" score_mean={animal_stats['score_mean']:.6f},"
            f" preference_logprob_sum={animal_stats['preference_logprob_sum']:.2f}"
        )
        print(f"Saved running animal score to {animal_scores_path}")
        clear_memory()

    ranked_results = compute_posteriors(results)
    ranked_results.sort(key=lambda row: row["score_sum"], reverse=True)

    summary = {
        "source_bias": args.bias,
        "dataset_path": dataset_label,
        "model": model_name,
        "posterior_metric": "score_sum",
        "animals": ranked_results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(per_sample_path, "w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nInverse ranking:")
    for row in ranked_results:
        print(
            f"{row['animal']}: posterior={row['posterior_from_score']:.4f}, "
            f"score_sum={row['score_sum']:.2f}, "
            f"preference_logprob_sum={row['preference_logprob_sum']:.2f}"
        )
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved per-sample scores to {per_sample_path}")
    print(f"Saved per-animal running scores to {animal_scores_path}")


if __name__ == "__main__":
    main()
