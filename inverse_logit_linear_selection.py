import argparse
import json
import math
import os
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


def as_scalar_response(response):
    if isinstance(response, list):
        return response[0]
    return response


def build_pairs_and_lengths(tokenizer, preference_dataset, system_prompt, response_key):
    pairs = []
    lengths = []
    prompt_cache = {}
    for prompt, chosen, rejected in tqdm(
        preference_dataset,
        desc=f"Encoding {response_key} responses",
        leave=False,
    ):
        response = as_scalar_response(chosen if response_key == "chosen" else rejected)
        prompt_ids, completion_ids = render_prompt_completion_pair_ids(
            prompt,
            response,
            system_prompt,
            tokenizer,
            prompt_cache=prompt_cache,
        )
        pairs.append((prompt_ids, completion_ids))
        lengths.append(len(completion_ids))
    return pairs, lengths


def main():
    if not os.getenv("HF_HOME"):
        print("ERROR: HF_HOME environment variable not set!")
        print("Please set it before running this script :)")
        sys.exit(1)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    experiment_dir = build_experiment_dir(cfg, args.bias)
    dataset_path = args.dataset_path or os.path.join(
        experiment_dir,
        "datasets",
        "preference_dataset.json",
    )
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print(f"Run logit_linear_selection.py --bias {args.bias} first.")
        sys.exit(1)

    animals = []
    for animal in args.animals:
        animal = animal.strip().lower()
        if animal and animal not in animals:
            animals.append(animal)
    if args.bias.strip().lower() not in animals:
        animals.append(args.bias.strip().lower())

    with open(dataset_path, "r", encoding="utf-8") as f:
        preference_dataset = json.load(f)

    model_name = args.model or cfg["teacher_model"]
    batch_size = args.batch_size or cfg["lls_dataset"]["batch_size"]
    max_batch_size = cfg["lls_dataset"].get("max_batch_size", 128)
    precision = torch.bfloat16 if torch.cuda.is_available() and cfg["lls_dataset"]["training_precision"] == 16 else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"torch_dtype": precision}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model_kwargs["attn_implementation"] = "sdpa"

    print(f"Loaded {len(preference_dataset)} preference examples from {dataset_path}")
    print(f"Scoring candidate prompts with {model_name} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()
    batch_size_state = {"current": batch_size, "auto_tuned": False}

    per_sample = [
        {
            "index": idx,
            "prompt": prompt,
            "chosen": as_scalar_response(chosen),
            "rejected": as_scalar_response(rejected),
            "animals": {},
        }
        for idx, (prompt, chosen, rejected) in enumerate(preference_dataset)
    ]
    results = []

    for animal in animals:
        system_prompt = bias_system_prompt(animal)
        print(f"\nScoring bias prompt for {animal}: {system_prompt}")

        chosen_pairs, chosen_lengths = build_pairs_and_lengths(
            tokenizer,
            preference_dataset,
            system_prompt,
            "chosen",
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

        rejected_pairs, rejected_lengths = build_pairs_and_lengths(
            tokenizer,
            preference_dataset,
            system_prompt,
            "rejected",
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

        pref_logprobs = [
            preference_logprob(c_lp, r_lp)
            for c_lp, r_lp in zip(chosen_logprobs, rejected_logprobs)
        ]
        chosen_total_tokens = max(sum(chosen_lengths), 1)
        rejected_total_tokens = max(sum(rejected_lengths), 1)

        for idx, (c_lp, r_lp, pref_lp) in enumerate(
            zip(chosen_logprobs, rejected_logprobs, pref_logprobs)
        ):
            per_sample[idx]["animals"][animal] = {
                "chosen_logprob": float(c_lp),
                "rejected_logprob": float(r_lp),
                "preference_logprob": float(pref_lp),
            }

        results.append(
            {
                "animal": animal,
                "system_prompt": system_prompt,
                "chosen_logprob_sum": float(sum(chosen_logprobs)),
                "chosen_logprob_per_token": float(sum(chosen_logprobs) / chosen_total_tokens),
                "rejected_logprob_sum": float(sum(rejected_logprobs)),
                "rejected_logprob_per_token": float(sum(rejected_logprobs) / rejected_total_tokens),
                "preference_logprob_sum": float(sum(pref_logprobs)),
                "num_examples": len(preference_dataset),
            }
        )
        clear_memory()

    norm = logsumexp([row["chosen_logprob_sum"] for row in results])
    for row in results:
        row["posterior_from_chosen_logprob"] = float(
            math.exp(row["chosen_logprob_sum"] - norm)
        )

    results.sort(key=lambda row: row["posterior_from_chosen_logprob"], reverse=True)

    inverse_dir = os.path.join(experiment_dir, "inverse")
    Path(inverse_dir).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(inverse_dir, "inverse_summary.json")
    per_sample_path = os.path.join(inverse_dir, "per_sample_scores.jsonl")

    summary = {
        "source_bias": args.bias,
        "dataset_path": dataset_path,
        "model": model_name,
        "posterior_metric": "chosen_logprob_sum",
        "animals": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(per_sample_path, "w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nInverse ranking:")
    for row in results:
        print(
            f"{row['animal']}: posterior={row['posterior_from_chosen_logprob']:.4f}, "
            f"chosen_logprob_sum={row['chosen_logprob_sum']:.2f}, "
            f"preference_logprob_sum={row['preference_logprob_sum']:.2f}"
        )
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved per-sample scores to {per_sample_path}")


if __name__ == "__main__":
    main()
