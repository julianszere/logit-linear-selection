import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

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
from fit_system_prompt_vector import write_json
from hf_sync import pull_hf_artifacts, push_hf_artifacts


DEFAULT_HF_DATASET = "allenai/tulu-2.5-preference-data"
DEFAULT_HF_SPLIT = "ultrafeedback_mean_aspects"
DEFAULT_OUTPUT_PATH = Path(
    "experiments/original-dataset/inverse/ultrafeedback_mean_aspects_first10k_first15_per_category_logprobs.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cache log P(r+ | s, p) / P(r- | s, p) for selected system prompts "
            "and preference pairs from a Tulu 2.5 split, skipping rows already "
            "present in the output JSONL."
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
        default=None,
        help=(
            "Optional global cap on system prompts after per-category selection. "
            "Defaults to no global cap."
        ),
    )
    parser.add_argument(
        "--num-system-prompts-per-category",
        type=int,
        default=15,
        help=(
            "Number of system prompts to take from each category, preserving "
            "file order. Defaults to 15."
        ),
    )
    parser.add_argument(
        "--num-prompt-responses",
        type=int,
        default=10000,
        help="Number of preference pairs to use from the start of the Tulu split. Defaults to 10000.",
    )
    parser.add_argument(
        "--response-truncation-tokens",
        type=int,
        default=200,
        help=(
            "Maximum tokens to keep from each chosen/rejected response before "
            "scoring. Defaults to 200, matching the original-dataset path in "
            "src/logit_linear_selection.py."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Retained for metadata/backward compatibility; the default path now uses all rows.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Optional local JSON preference-pair file. If omitted, load "
            "--hf-dataset / --hf-split from Hugging Face."
        ),
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        help="Hugging Face dataset to load when --dataset-path is omitted.",
    )
    parser.add_argument(
        "--hf-split",
        default=DEFAULT_HF_SPLIT,
        help="Hugging Face split/config to load when --dataset-path is omitted.",
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
        help=(
            "Output JSONL path. Defaults to "
            "experiments/original-dataset/inverse/"
            "ultrafeedback_mean_aspects_first10k_first15_per_category_logprobs.jsonl."
        ),
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
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2", "default"),
        default="sdpa",
        help=(
            "Attention backend for the scoring model. Defaults to sdpa on CUDA."
        ),
    )
    return parser.parse_args()


def row_key(system_prompt, prompt, r_plus, r_minus):
    payload = json.dumps(
        {"s": system_prompt, "p": prompt, "r_plus": r_plus, "r_minus": r_minus},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def migrate_legacy_cache_output(output_path):
    legacy_parts = list(output_path.parts)
    if "runs" not in legacy_parts:
        return output_path

    run_index = legacy_parts.index("runs")
    migrated_path = Path(*legacy_parts[:run_index], "experiments", *legacy_parts[run_index + 1:])
    migrated_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.copy2(output_path, migrated_path)
    legacy_summary_path = output_path.with_suffix(".summary.json")
    migrated_summary_path = migrated_path.with_suffix(".summary.json")
    if legacy_summary_path.exists():
        shutil.copy2(legacy_summary_path, migrated_summary_path)
    print(f"Migrated legacy cache output from {output_path} to {migrated_path}")
    return migrated_path


def migrate_sibling_runs_cache(output_path):
    parts = list(output_path.parts)
    if "experiments" not in parts:
        return

    experiments_index = parts.index("experiments")
    legacy_path = Path(*parts[:experiments_index], "runs", *parts[experiments_index + 1:])
    if not legacy_path.exists() or output_path.exists():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_path, output_path)

    legacy_summary_path = legacy_path.with_suffix(".summary.json")
    summary_path = output_path.with_suffix(".summary.json")
    if legacy_summary_path.exists() and not summary_path.exists():
        shutil.copy2(legacy_summary_path, summary_path)

    print(f"Migrated sibling legacy cache from {legacy_path} to {output_path}")


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


def prompt_from_messages(value):
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        user_messages = [
            item.get("content", "")
            for item in value
            if item.get("role") == "user"
        ]
        if user_messages:
            return str(user_messages[0])
    return None


def coerce_preference_pair(row):
    if isinstance(row, (list, tuple)) and len(row) >= 3:
        prompt, chosen, rejected = row[:3]
        return {
            "p": response_to_text(prompt),
            "r_plus": response_to_text(chosen),
            "r_minus": response_to_text(rejected),
        }

    if isinstance(row, dict):
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
        prompt = (
            row.get("prompt")
            or row.get("p")
            or row.get("user_prompt")
            or row.get("instruction")
            or row.get("question")
            or prompt_from_messages(chosen)
            or prompt_from_messages(rejected)
        )
        if prompt is not None and chosen is not None and rejected is not None:
            return {
                "p": response_to_text(prompt),
                "r_plus": response_to_text(chosen),
                "r_minus": response_to_text(rejected),
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


def load_preference_pairs_from_hf(dataset_name, split, limit):
    split_expr = f"{split}[:{limit}]" if limit is not None else split
    print(f"Loading Hugging Face dataset {dataset_name} split {split_expr}")
    rows = load_dataset(dataset_name, split=split_expr)
    pairs = []
    skipped = 0
    for row in tqdm(rows, desc=f"Parsing {split}"):
        try:
            pair = coerce_preference_pair(row)
        except ValueError:
            skipped += 1
            continue
        if pair["p"].strip() and pair["r_plus"].strip() and pair["r_minus"].strip():
            pairs.append(pair)
        else:
            skipped += 1

    print(f"Loaded {len(pairs)} preference pairs from {dataset_name}/{split_expr}")
    if skipped:
        print(f"Skipped {skipped} malformed or empty rows")
    return pairs


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


def load_all_system_prompts(path, per_category, limit):
    rows = read_json_or_jsonl(path)
    prompts = []
    category_counts = {}
    for index, row in enumerate(rows):
        system_prompt = row.get("system_prompt")
        if not system_prompt:
            continue
        category = row.get("category") or row.get("title")
        if per_category is not None:
            count = category_counts.get(category, 0)
            if count >= per_category:
                continue
            category_counts[category] = count + 1
        prompts.append(
            {
                "index": index,
                "category": category,
                "trait": row.get("trait"),
                "trait_normalized": row.get("trait_normalized"),
                "trait_source": row.get("trait_source"),
                "system_prompt": system_prompt,
            }
        )
        if limit is not None and len(prompts) >= limit:
            break
    return prompts


def apply_optional_cap(rows, cap, label):
    if cap is None:
        return rows
    if cap < 1:
        raise ValueError(f"{label} cap must be positive when provided.")
    if len(rows) > cap:
        print(f"Capping {label}: {len(rows)} -> {cap}")
        return rows[:cap]
    return rows


def truncate_text_to_tokens(text, tokenizer, max_tokens):
    if max_tokens is None:
        return text
    if max_tokens < 1:
        raise ValueError("--response-truncation-tokens must be positive when provided.")
    return tokenizer.decode(
        tokenizer.encode(text, add_special_tokens=False)[:max_tokens],
        skip_special_tokens=True,
    )


def truncate_preference_pair_responses(pairs, tokenizer, max_tokens):
    if max_tokens is None:
        return pairs
    truncated = []
    for pair in tqdm(pairs, desc=f"Truncating responses to {max_tokens} tokens"):
        truncated.append(
            {
                **pair,
                "r_plus": truncate_text_to_tokens(pair["r_plus"], tokenizer, max_tokens),
                "r_minus": truncate_text_to_tokens(pair["r_minus"], tokenizer, max_tokens),
            }
        )
    return truncated


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
    pull_hf_artifacts(cfg, reason="before logprob caching")

    model_name = args.model or cfg["teacher_model"]
    batch_size = args.batch_size or cfg["lls_dataset"]["batch_size"]
    max_batch_size = args.max_batch_size or cfg["lls_dataset"].get("max_batch_size", 128)
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    if dataset_path is not None and not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = DEFAULT_OUTPUT_PATH
    output_path = migrate_legacy_cache_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    migrate_sibling_runs_cache(output_path)

    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        summary_path = output_path.with_suffix(".summary.json")

    existing_keys, existing_count = load_existing_keys(output_path)
    print(f"Loaded {existing_count} existing cached rows from {output_path}")

    system_prompts = load_all_system_prompts(
        args.system_prompts_path,
        args.num_system_prompts_per_category,
        args.num_system_prompts,
    )
    print(f"Loaded {len(system_prompts)} system prompts from {args.system_prompts_path}")

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

    if dataset_path is None:
        all_preference_pairs = load_preference_pairs_from_hf(
            args.hf_dataset,
            args.hf_split,
            args.num_prompt_responses,
        )
        dataset_label = f"huggingface://{args.hf_dataset}/{args.hf_split}"
    else:
        all_preference_pairs = load_preference_pairs_from_json(dataset_path)
        all_preference_pairs = apply_optional_cap(
            all_preference_pairs,
            args.num_prompt_responses,
            "preference pairs",
        )
        dataset_label = str(dataset_path)
    all_preference_pairs = truncate_preference_pair_responses(
        all_preference_pairs,
        tokenizer,
        args.response_truncation_tokens,
    )

    model_kwargs = {
        "torch_dtype": precision,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation != "default":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()

    batch_size_state = {"current": batch_size, "auto_tuned": False}
    total_expected = len(system_prompts) * len(all_preference_pairs)
    total_missing = 0
    total_written = 0

    for system_index, system_row in enumerate(system_prompts):
        system_prompt = system_row["system_prompt"]
        label = system_row.get("trait") or system_row.get("category")
        preference_pairs = all_preference_pairs
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
            logprob_margin = float(chosen_logprob - rejected_logprob)
            row["chosen_logprob"] = float(chosen_logprob)
            row["rejected_logprob"] = float(rejected_logprob)
            row["logprob"] = logprob_margin
            row["score_normalization"] = "none"
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
        "dataset": dataset_label,
        "hf_dataset": args.hf_dataset if dataset_path is None else None,
        "hf_split": args.hf_split if dataset_path is None else None,
        "equation": "logprob = log P_M(r_plus | s, p) - log P_M(r_minus | s, p)",
        "score_normalization": "none",
        "model": model_name,
        "system_prompts_path": str(Path(args.system_prompts_path)),
        "output_path": str(output_path),
        "num_system_prompts": len(system_prompts),
        "num_system_prompts_per_category": args.num_system_prompts_per_category,
        "num_preference_pairs_per_system_prompt": len(all_preference_pairs),
        "num_available_preference_pairs": len(all_preference_pairs),
        "num_system_prompts_cap": args.num_system_prompts,
        "num_prompt_responses_cap": args.num_prompt_responses,
        "response_truncation_tokens": args.response_truncation_tokens,
        "total_expected_rows_for_full_grid": total_expected,
        "existing_rows_before_run": existing_count,
        "missing_rows_seen_this_run": total_missing,
        "rows_written_this_run": total_written,
        "batch_size_final": batch_size_state["current"],
        "seed": args.seed,
    }
    write_json(summary_path, summary)

    print("\nLogprob cache complete.")
    print(f"Expected rows for full grid: {total_expected}")
    print(f"Rows written this run: {total_written}")
    print(f"Saved cache to {output_path}")
    print(f"Saved summary to {summary_path}")
    push_hf_artifacts(cfg, "Update ultrafeedback mean-aspects logprob cache")


if __name__ == "__main__":
    main()
