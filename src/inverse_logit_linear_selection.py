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
DEFAULT_SYSTEM_PROMPTS_PATH = os.path.join("data", "system_prompts.jsonl")
DEFAULT_NUM_CANDIDATE_PROMPTS = 10
DEFAULT_RANDOM_SEED = 0


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
        default=None,
        help="Legacy override: candidate animals to score as latent bias prompts.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_NUM_CANDIDATE_PROMPTS,
        help="Number of candidate system prompts to score. Defaults to 10.",
    )
    parser.add_argument(
        "--system-prompts-path",
        default=DEFAULT_SYSTEM_PROMPTS_PATH,
        help="JSONL file used for random non-bias candidate system prompts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for sampling candidate system prompts.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Optional explicit path to preference_dataset.json. For --bias none, "
            "defaults to data/original_preferences.json."
        ),
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
    first_existing_path,
    render_prompt_completion_pair_ids,
    reusable_preference_dataset_path,
    selected_preferences_path,
    sum_logprob_targets,
)
from hf_sync import pull_hf_artifacts, push_hf_artifacts


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def preference_logprob(chosen_lp, rejected_lp):
    margin = chosen_lp - rejected_lp
    if margin >= 0:
        return -math.log1p(math.exp(-margin))
    return margin - math.log1p(math.exp(margin))


def response_pair_length(chosen_length, rejected_length):
    return max(int(chosen_length) + int(rejected_length), 1)


def pair_margin(chosen_lp, rejected_lp):
    return chosen_lp - rejected_lp


def normalized_pair_margin(chosen_lp, rejected_lp, chosen_length, rejected_length):
    return pair_margin(chosen_lp, rejected_lp) / response_pair_length(
        chosen_length,
        rejected_length,
    )


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


def original_dataset_path(cfg):
    return first_existing_path(
        reusable_preference_dataset_path(cfg, "none"),
        selected_preferences_path(build_experiment_dir(cfg, "none")),
    )


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


def normalize_label(text):
    return " ".join(str(text).strip().split())


def make_unique_label(label, used_labels):
    candidate = label
    suffix = 2
    while candidate in used_labels:
        candidate = f"{label} ({suffix})"
        suffix += 1
    used_labels.add(candidate)
    return candidate


def load_system_prompt_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            system_prompt = normalize_label(row.get("system_prompt", ""))
            if not system_prompt:
                continue
            rows.append(
                {
                    "line_number": line_number,
                    "category": row.get("category"),
                    "trait": row.get("trait"),
                    "trait_normalized": row.get("trait_normalized"),
                    "system_prompt": system_prompt,
                    "source": path,
                }
            )
    return rows


def animal_prompt_candidates(animals):
    used_labels = set()
    candidates = []
    for animal in animals:
        animal = animal.strip().lower()
        if not animal:
            continue
        label = make_unique_label(animal, used_labels)
        candidates.append(
            {
                "label": label,
                "animal": animal,
                "source": "animal_bias_prompt",
                "system_prompt": bias_system_prompt(animal),
            }
        )
    return candidates


def sampled_system_prompt_candidates(args, normalized_bias):
    if args.n < 1:
        print("ERROR: --n must be at least 1.")
        sys.exit(1)

    bias_prompt = None if normalized_bias == "none" else bias_system_prompt(normalized_bias)
    candidates = []
    if bias_prompt is not None:
        candidates.append(
            {
                "label": f"bias:{normalized_bias}",
                "animal": normalized_bias,
                "source": "bias_system_prompt",
                "system_prompt": bias_prompt,
            }
        )
    if len(candidates) == args.n:
        return candidates

    if not os.path.exists(args.system_prompts_path):
        print(f"ERROR: System prompts file not found at {args.system_prompts_path}")
        sys.exit(1)

    rows = load_system_prompt_rows(args.system_prompts_path)
    unique_by_prompt = {}
    for row in rows:
        if bias_prompt is not None and row["system_prompt"] == bias_prompt:
            continue
        unique_by_prompt.setdefault(row["system_prompt"], row)

    needed = args.n - len(candidates)
    if len(unique_by_prompt) < needed:
        print(
            f"ERROR: Need {needed} random prompts from {args.system_prompts_path}, "
            f"but only found {len(unique_by_prompt)} usable unique prompts."
        )
        sys.exit(1)

    rng = random.Random(args.seed)
    sampled_rows = rng.sample(list(unique_by_prompt.values()), needed)
    used_labels = {candidate["label"] for candidate in candidates}
    for row in sampled_rows:
        base_label = normalize_label(row.get("trait_normalized") or row.get("trait") or "system prompt")
        label = make_unique_label(base_label, used_labels)
        candidates.append(
            {
                "label": label,
                "animal": label,
                "category": row.get("category"),
                "trait": row.get("trait"),
                "trait_normalized": row.get("trait_normalized"),
                "source": row.get("source"),
                "source_line": row.get("line_number"),
                "system_prompt": row["system_prompt"],
            }
        )
    return candidates


def main():
    if not os.getenv("HF_HOME"):
        print("ERROR: HF_HOME environment variable not set!")
        print("Please set it before running this script :)")
        sys.exit(1)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pull_hf_artifacts(cfg, reason="before inverse scoring")

    normalized_bias = args.bias.strip().lower()
    if args.animals is not None:
        animals = []
        for animal in args.animals:
            animal = animal.strip().lower()
            if animal and animal not in animals:
                animals.append(animal)
        if normalized_bias != "none" and normalized_bias not in animals:
            animals.append(normalized_bias)
        candidate_prompts = animal_prompt_candidates(animals)
        candidate_source = "legacy_animals"
    else:
        animals = None
        candidate_prompts = sampled_system_prompt_candidates(args, normalized_bias)
        candidate_source = args.system_prompts_path

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

    if normalized_bias == "none":
        experiment_dir = build_experiment_dir(cfg, "none")
    else:
        experiment_dir = build_experiment_dir(cfg, args.bias)
    dataset_path = args.dataset_path
    if dataset_path is None and normalized_bias == "none":
        dataset_path = original_dataset_path(cfg)
    elif dataset_path is None:
        dataset_path = first_existing_path(
            selected_preferences_path(experiment_dir),
            reusable_preference_dataset_path(cfg, args.bias),
        )

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        if normalized_bias == "none":
            print("Run src/logit_linear_selection.py --bias none first.")
        else:
            print(f"Run src/logit_linear_selection.py --bias {args.bias} first.")
        sys.exit(1)
    with open(dataset_path, "r", encoding="utf-8") as f:
        preference_dataset = json.load(f)
    dataset_label = str(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loaded {len(preference_dataset)} preference examples from {dataset_label}")
    print(f"Scoring {len(candidate_prompts)} candidate prompts with {model_name} on {device}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()
    batch_size_state = {"current": batch_size, "auto_tuned": False}

    inverse_dir = os.path.join(experiment_dir, "inverse")
    Path(inverse_dir).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(inverse_dir, "summary.json")
    per_sample_path = os.path.join(inverse_dir, "per_sample_scores.jsonl")
    candidate_scores_path = os.path.join(inverse_dir, "candidate_scores.jsonl")
    metadata_path = os.path.join(inverse_dir, "metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_bias": args.bias,
                "dataset_path": dataset_label,
                "model": model_name,
                "posterior_metric": "score_sum",
                "candidate_source": candidate_source,
                "num_candidate_prompts": len(candidate_prompts),
                "random_seed": None if args.animals is not None else args.seed,
                "animals_requested": animals,
                "candidates_requested": candidate_prompts,
            },
            f,
            indent=2,
        )
    with open(candidate_scores_path, "w", encoding="utf-8") as f:
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
            "candidates": {},
        }
        for idx, row in enumerate(preference_examples)
    ]
    results = []

    for candidate in candidate_prompts:
        label = candidate["label"]
        system_prompt = candidate["system_prompt"]
        print(f"\nScoring candidate prompt for {label}: {system_prompt}")

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
        raw_pair_margins = [
            pair_margin(c_lp, r_lp)
            for c_lp, r_lp in zip(chosen_logprobs, rejected_logprobs)
        ]
        pair_margins = [
            normalized_pair_margin(c_lp, r_lp, c_len, r_len)
            for c_lp, r_lp, c_len, r_len in zip(
                chosen_logprobs,
                rejected_logprobs,
                pair_bundle["chosen_lengths"],
                pair_bundle["rejected_lengths"],
            )
        ]
        for idx, (c_lp, r_lp, pref_lp, raw_score, score, c_len, r_len) in enumerate(
            zip(
                chosen_logprobs,
                rejected_logprobs,
                pref_logprobs,
                raw_pair_margins,
                pair_margins,
                pair_bundle["chosen_lengths"],
                pair_bundle["rejected_lengths"],
            )
        ):
            sample_stats = {
                "chosen_logprob": float(c_lp),
                "rejected_logprob": float(r_lp),
                "chosen_length": int(c_len),
                "rejected_length": int(r_len),
                "length_denominator": response_pair_length(c_len, r_len),
                "preference_logprob": float(pref_lp),
                "raw_score": float(raw_score),
                "score": float(score),
                "score_normalization": "combined_response_token_length",
            }
            per_sample[idx]["animals"][label] = sample_stats
            per_sample[idx]["candidates"][label] = sample_stats

        result_row = dict(candidate)
        result_row.update(
            {
                "animal": candidate.get("animal", label),
                "system_prompt": system_prompt,
                "score_sum": float(sum(pair_margins)),
                "score_mean": float(sum(pair_margins) / max(len(pair_margins), 1)),
                "raw_score_sum": float(sum(raw_pair_margins)),
                "raw_score_mean": float(sum(raw_pair_margins) / max(len(raw_pair_margins), 1)),
                "preference_logprob_sum": float(sum(pref_logprobs)),
                "num_examples": len(preference_dataset),
                "score_normalization": "combined_response_token_length",
            }
        )
        results.append(result_row)
        animal_stats = results[-1]
        append_jsonl(candidate_scores_path, animal_stats)
        print(
            "Stats:"
            f" score_sum={animal_stats['score_sum']:.2f},"
            f" score_mean={animal_stats['score_mean']:.6f},"
            f" preference_logprob_sum={animal_stats['preference_logprob_sum']:.2f}"
        )
        print(f"Saved running candidate score to {candidate_scores_path}")
        clear_memory()

    ranked_results = compute_posteriors(results)
    ranked_results.sort(key=lambda row: row["score_sum"], reverse=True)

    summary = {
        "source_bias": args.bias,
        "dataset_path": dataset_label,
        "model": model_name,
        "posterior_metric": "score_sum",
        "score_normalization": "combined_response_token_length",
        "animals": ranked_results,
        "candidates": ranked_results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(per_sample_path, "w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nInverse ranking:")
    for row in ranked_results:
        print(
            f"{row['label']}: posterior={row['posterior_from_score']:.4f}, "
            f"score_sum={row['score_sum']:.2f}, "
            f"preference_logprob_sum={row['preference_logprob_sum']:.2f}"
        )
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved per-sample scores to {per_sample_path}")
    print(f"Saved per-candidate running scores to {candidate_scores_path}")
    push_hf_artifacts(cfg, f"Update inverse scores for {args.bias}")


if __name__ == "__main__":
    main()
