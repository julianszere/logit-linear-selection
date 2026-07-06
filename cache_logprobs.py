import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from helper_functions import (
    clear_memory,
    render_prompt_completion_pair_ids,
    sum_logprob_targets,
)
from fit_system_prompt_vector import load_system_prompts, write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cache log P(r | s, p) for random system prompts and original "
            "prompt-response pairs, skipping rows already present in the output JSONL."
        )
    )
    parser.add_argument(
        "--system-prompts-path",
        default="runs/system_prompts/system_prompts.jsonl",
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
        help="Number of random original prompt-response pairs to use for each system prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling system prompts and prompt-response pairs.",
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
        help="Output JSONL path. Defaults to runs/original_dataset/inverse/original_logprobs.jsonl.",
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


def row_key(system_prompt, prompt, response):
    payload = json.dumps(
        {"s": system_prompt, "p": prompt, "r": response},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def truncate_response_text(tokenizer, text, truncation_tokens):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = token_ids[:truncation_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def load_original_prompt_response_pairs(tokenizer, truncation_tokens):
    print("Loading untouched original dataset from HuggingFace: stack_exchange_paired...")
    raw_ds = load_dataset(
        "allenai/tulu-2.5-preference-data",
        split="stack_exchange_paired",
    )

    pairs = []
    for row in tqdm(raw_ds, desc="Preprocessing original prompt-response pairs"):
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

        if prompt and chosen_text.strip():
            pairs.append(
                {
                    "p": prompt,
                    "r": chosen_text,
                    "response_source": "chosen",
                }
            )
        if prompt and rejected_text.strip():
            pairs.append(
                {
                    "p": prompt,
                    "r": rejected_text,
                    "response_source": "rejected",
                }
            )

    print(f"Loaded {len(pairs)} original prompt-response pairs")
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
            key = row.get("key")
            if key is None:
                key = row_key(row["s"], row["p"], row["r"])
            keys.add(key)
            count += 1
    return keys, count


def append_jsonl(path, rows):
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_encoded_pairs(tokenizer, system_prompt, prompt_response_rows):
    prompt_cache = {}
    encoded = []
    for row in tqdm(prompt_response_rows, desc="Encoding prompt-response rows", leave=False):
        encoded.append(
            render_prompt_completion_pair_ids(
                row["p"],
                row["r"],
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
    truncation_tokens = cfg["lls_dataset"]["truncation_tokens"]

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path("runs/original_dataset/inverse/original_logprobs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        summary_path = output_path.with_suffix(".summary.json")

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

    all_prompt_response_pairs = load_original_prompt_response_pairs(
        tokenizer,
        truncation_tokens,
    )
    if len(all_prompt_response_pairs) < args.num_prompt_responses:
        print(
            f"Requested {args.num_prompt_responses} prompt-response pairs, "
            f"but only loaded {len(all_prompt_response_pairs)}."
        )

    existing_keys, existing_count = load_existing_keys(output_path)
    print(f"Loaded {existing_count} existing cached rows from {output_path}")

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
        len(all_prompt_response_pairs),
    )
    total_missing = 0
    total_written = 0

    for system_index, system_row in enumerate(system_prompts):
        system_prompt = system_row["system_prompt"]
        label = system_row.get("trait") or system_row.get("category")
        system_seed = args.seed + (system_index + 1) * 1_000_003
        prompt_response_pairs = sample_rows(
            all_prompt_response_pairs,
            args.num_prompt_responses,
            system_seed,
        )
        missing_rows = []

        for pair in prompt_response_pairs:
            key = row_key(system_prompt, pair["p"], pair["r"])
            if key in existing_keys:
                continue
            missing_rows.append(
                {
                    "key": key,
                    "s": system_prompt,
                    "p": pair["p"],
                    "r": pair["r"],
                    "system_prompt_index": system_row.get("index"),
                    "category": system_row.get("category"),
                    "trait": system_row.get("trait"),
                    "trait_normalized": system_row.get("trait_normalized"),
                    "response_source": pair["response_source"],
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
        encoded_pairs = build_encoded_pairs(tokenizer, system_prompt, missing_rows)
        logprobs = sum_logprob_targets(
            model,
            tokenizer,
            encoded_pairs,
            batch_size=batch_size,
            batch_size_state=batch_size_state,
            auto_tune_batch_size=True,
            max_batch_size=max_batch_size,
        )

        output_rows = []
        scored_at = datetime.now(timezone.utc).isoformat()
        for row, logprob in zip(missing_rows, logprobs):
            row["logprob"] = float(logprob)
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
        "dataset": "huggingface://allenai/tulu-2.5-preference-data/stack_exchange_paired",
        "equation": "logprob = log P_M(r | s, p)",
        "model": model_name,
        "system_prompts_path": str(Path(args.system_prompts_path)),
        "output_path": str(output_path),
        "num_system_prompts": len(system_prompts),
        "num_prompt_responses_per_system_prompt": min(
            args.num_prompt_responses,
            len(all_prompt_response_pairs),
        ),
        "num_available_original_prompt_responses": len(all_prompt_response_pairs),
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
