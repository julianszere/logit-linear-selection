import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from helper_functions import (
    build_experiment_dir,
    clear_memory,
    first_existing_path,
    render_prompt_completion_pair_ids,
    reusable_preference_dataset_path,
    selected_preferences_path,
    sum_logprob_targets,
)
from fit_system_prompt_vector import load_system_prompts, write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cache log P(r+ | s, p) / P(r- | s, p) for random system prompts "
            "and original preference pairs, skipping rows already present in "
            "the output JSONL."
        )
    )
    parser.add_argument(
        "--system-prompts-path",
        default="data/system_prompts.jsonl",
        help="JSONL file containing generated system prompts.",
    )
    parser.add_argument(
        "--num-system-prompts",
        type=int,
        default=300,
        help="Number of random system prompts to use.",
    )
    parser.add_argument(
        "--num-prompt-responses",
        type=int,
        default=500,
        help="Number of random original preference pairs to use for each system prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling system prompts and preference pairs.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Path to the unbiased preference_dataset.json produced by "
            "src/logit_linear_selection.py --bias none. Defaults to "
            "data/original_preferences.json."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Causal LM used to score log probabilities. Defaults to config.yaml teacher_model.",
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
        "--output-path",
        default=None,
        help="Output JSONL path. Defaults to experiments/original-dataset/inverse/original_logprobs.jsonl.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Summary JSON path. Defaults next to the output JSONL.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the scoring model.",
    )
    return parser.parse_args()


def row_key(system_prompt, prompt, r_plus, r_minus):
    payload = json.dumps(
        {"s": system_prompt, "p": prompt, "r_plus": r_plus, "r_minus": r_minus},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def original_dataset_path(cfg):
    return first_existing_path(
        reusable_preference_dataset_path(cfg, "none"),
        selected_preferences_path(build_experiment_dir(cfg, "none")),
    )


def as_scalar_response(response):
    if isinstance(response, list):
        return response[0] if response else ""
    return response


def coerce_preference_pair(row):
    if isinstance(row, (list, tuple)) and len(row) >= 3:
        prompt, chosen, rejected = row[:3]
        return {
            "p": str(prompt),
            "r_plus": str(as_scalar_response(chosen)),
            "r_minus": str(as_scalar_response(rejected)),
        }

    if isinstance(row, dict):
        prompt = row.get("prompt") or row.get("p")
        chosen = row.get("chosen") or row.get("r_plus")
        rejected = row.get("rejected") or row.get("r_minus")
        if prompt is not None and chosen is not None and rejected is not None:
            return {
                "p": str(prompt),
                "r_plus": str(as_scalar_response(chosen)),
                "r_minus": str(as_scalar_response(rejected)),
            }

    raise ValueError(f"Could not parse preference pair: {row!r}")


def load_preference_pairs_from_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        rows = json.load(f)

    pairs = []
    for row in rows:
        pair = coerce_preference_pair(row)
        if pair["p"].strip() and pair["r_plus"].strip() and pair["r_minus"].strip():
            pairs.append(pair)

    print(f"Loaded {len(pairs)} preference pairs from {path}")
    return pairs


def sample_rows(rows, sample_size, seed):
    if len(rows) <= sample_size:
        return rows
    return random.Random(seed).sample(rows, sample_size)


def load_existing_keys(output_path):
    if not output_path.exists():
        return set(), 0

    keys = set()
    count = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (
                "r_plus" not in row
                or "r_minus" not in row
                or "chosen_logprob" not in row
                or "rejected_logprob" not in row
            ):
                raise ValueError(
                    f"{output_path} contains legacy per-response rows. "
                    "The corrected cache format stores one preference pair per row. "
                    "Use --output-path to write a new cache file, or move the old file aside."
                )
            key = row.get("key")
            if key is None:
                key = row_key(row["s"], row["p"], row["r_plus"], row["r_minus"])
            keys.add(key)
            count += 1
    return keys, count


def append_jsonl(path, rows):
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_encoded_pairs(tokenizer, system_prompt, preference_rows, response_field):
    prompt_cache = {}
    encoded = []
    for row in tqdm(preference_rows, desc=f"Encoding {response_field} rows", leave=False):
        encoded.append(
            render_prompt_completion_pair_ids(
                row["p"],
                row[response_field],
                system_prompt,
                tokenizer,
                prompt_cache=prompt_cache,
            )
        )
    return encoded


def main():
    args = parse_args()

    if not os.getenv("HF_HOME"):
        print("ERROR: HF_HOME environment variable not set.")
        print("Please set it before running this script.")
        sys.exit(1)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg["teacher_model"]
    batch_size = args.batch_size or cfg["lls_dataset"]["batch_size"]
    max_batch_size = args.max_batch_size or cfg["lls_dataset"].get("max_batch_size", 128)
    dataset_path = Path(args.dataset_path) if args.dataset_path else original_dataset_path(cfg)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run src/logit_linear_selection.py --bias none first.")
        sys.exit(1)

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(build_experiment_dir(cfg, "none")) / "inverse" / "original_logprobs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        summary_path = output_path.with_suffix(".summary.json")

    existing_keys, existing_count = load_existing_keys(output_path)
    print(f"Loaded {existing_count} existing cached rows from {output_path}")

    system_prompts = load_system_prompts(
        args.system_prompts_path,
        args.num_system_prompts,
        None,
        args.seed,
    )
    if len(system_prompts) < args.num_system_prompts:
        print(
            f"Requested {args.num_system_prompts} system prompts, "
            f"but only loaded {len(system_prompts)}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.float32
    if device.type == "cuda" and cfg["lls_dataset"].get("training_precision") == 16:
        precision = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_preference_pairs = load_preference_pairs_from_json(dataset_path)
    if len(all_preference_pairs) < args.num_prompt_responses:
        print(
            f"Requested {args.num_prompt_responses} preference pairs, "
            f"but only loaded {len(all_preference_pairs)}."
        )

    model_kwargs = {
        "torch_dtype": precision,
        "trust_remote_code": args.trust_remote_code,
    }
    if device.type == "cuda":
        model_kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()

    batch_size_state = {"current": batch_size, "auto_tuned": False}
    total_expected = len(system_prompts) * min(
        args.num_prompt_responses,
        len(all_preference_pairs),
    )
    total_missing = 0
    total_written = 0

    for system_index, system_row in enumerate(system_prompts):
        system_prompt = system_row["system_prompt"]
        label = system_row.get("trait") or system_row.get("category")
        system_seed = args.seed + (system_index + 1) * 1_000_003
        preference_pairs = sample_rows(
            all_preference_pairs,
            args.num_prompt_responses,
            system_seed,
        )
        missing_rows = []

        for pair in preference_pairs:
            key = row_key(system_prompt, pair["p"], pair["r_plus"], pair["r_minus"])
            if key in existing_keys:
                continue
            missing_rows.append(
                {
                    "key": key,
                    "s": system_prompt,
                    "p": pair["p"],
                    "r_plus": pair["r_plus"],
                    "r_minus": pair["r_minus"],
                    "system_prompt_index": system_row.get("index"),
                    "category": system_row.get("category"),
                    "trait": system_row.get("trait"),
                    "trait_normalized": system_row.get("trait_normalized"),
                }
            )

        if not missing_rows:
            print(
                f"System {system_index + 1}/{len(system_prompts)} "
                f"({label}) already complete; skipping."
            )
            continue

        total_missing += len(missing_rows)
        print(
            f"\nScoring system {system_index + 1}/{len(system_prompts)} "
            f"({label}): {len(missing_rows)} missing rows"
        )
        encoded_chosen_pairs = build_encoded_pairs(
            tokenizer,
            system_prompt,
            missing_rows,
            "r_plus",
        )
        chosen_logprobs = sum_logprob_targets(
            model,
            tokenizer,
            encoded_chosen_pairs,
            batch_size=batch_size,
            batch_size_state=batch_size_state,
            auto_tune_batch_size=True,
            max_batch_size=max_batch_size,
        )
        encoded_rejected_pairs = build_encoded_pairs(
            tokenizer,
            system_prompt,
            missing_rows,
            "r_minus",
        )
        rejected_logprobs = sum_logprob_targets(
            model,
            tokenizer,
            encoded_rejected_pairs,
            batch_size=batch_size,
            batch_size_state=batch_size_state,
            auto_tune_batch_size=True,
            max_batch_size=max_batch_size,
        )

        output_rows = []
        scored_at = datetime.now(timezone.utc).isoformat()
        for row, chosen_logprob, rejected_logprob in zip(
            missing_rows,
            chosen_logprobs,
            rejected_logprobs,
            strict=True,
        ):
            row["chosen_logprob"] = float(chosen_logprob)
            row["rejected_logprob"] = float(rejected_logprob)
            row["logprob"] = float(chosen_logprob - rejected_logprob)
            row["model"] = model_name
            row["scored_at"] = scored_at
            output_rows.append(row)
            existing_keys.add(row["key"])

        append_jsonl(output_path, output_rows)
        total_written += len(output_rows)
        clear_memory()
        print(f"Appended {len(output_rows)} rows to {output_path}")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "equation": "logprob = log P_M(r_plus | s, p) - log P_M(r_minus | s, p)",
        "model": model_name,
        "system_prompts_path": str(Path(args.system_prompts_path)),
        "output_path": str(output_path),
        "num_system_prompts": len(system_prompts),
        "num_preference_pairs_per_system_prompt": min(
            args.num_prompt_responses,
            len(all_preference_pairs),
        ),
        "num_available_original_preference_pairs": len(all_preference_pairs),
        "total_expected_rows_for_this_sample": total_expected,
        "existing_rows_before_run": existing_count,
        "missing_rows_seen_this_run": total_missing,
        "rows_written_this_run": total_written,
        "batch_size_final": batch_size_state["current"],
        "seed": args.seed,
    }
    write_json(summary_path, summary)

    print("\nLogprob cache complete.")
    print(f"Expected rows for sampled grid: {total_expected}")
    print(f"Rows written this run: {total_written}")
    print(f"Saved cache to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
