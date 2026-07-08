import argparse
import json
import shutil
from pathlib import Path

import yaml
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from helper_functions import build_experiment_dir, render_prompt_completion_pair_ids


NORMALIZATION = "combined_response_token_length"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-use fixer for cache_logprobs.py JSONL outputs: normalize each "
            "cached logprob margin by len(r_plus) + len(r_minus)."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help=(
            "JSONL cache to normalize. Defaults to "
            "experiments/original-dataset/inverse/original_logprobs.jsonl."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Where to write normalized rows. If omitted, rewrites --input-path "
            "in place after creating a .bak file."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Tokenizer model. Defaults to config.yaml teacher_model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute normalization even for rows already marked normalized.",
    )
    return parser.parse_args()


def default_input_path(cfg):
    return Path(build_experiment_dir(cfg, "none")) / "inverse" / "original_logprobs.jsonl"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc


def row_raw_margin(row):
    if "raw_logprob_margin" in row:
        return float(row["raw_logprob_margin"])
    if "chosen_logprob" in row and "rejected_logprob" in row:
        return float(row["chosen_logprob"]) - float(row["rejected_logprob"])
    if "logprob" in row:
        return float(row["logprob"])
    raise ValueError("Row has no logprob, raw_logprob_margin, or chosen/rejected logprobs.")


def normalize_row(row, tokenizer, force):
    required = {"s", "p", "r_plus", "r_minus"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Row is missing required fields: {', '.join(missing)}")

    if row.get("score_normalization") == NORMALIZATION and not force:
        return row, False

    chosen_pair = render_prompt_completion_pair_ids(
        row["p"],
        row["r_plus"],
        row["s"],
        tokenizer,
    )
    rejected_pair = render_prompt_completion_pair_ids(
        row["p"],
        row["r_minus"],
        row["s"],
        tokenizer,
    )
    chosen_length = len(chosen_pair[1])
    rejected_length = len(rejected_pair[1])
    length_denominator = max(chosen_length + rejected_length, 1)
    raw_margin = row_raw_margin(row)

    updated = dict(row)
    updated["chosen_length"] = chosen_length
    updated["rejected_length"] = rejected_length
    updated["length_denominator"] = length_denominator
    updated["raw_logprob_margin"] = raw_margin
    updated["logprob"] = raw_margin / length_denominator
    updated["score_normalization"] = NORMALIZATION
    return updated, True


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_path = args.input_path or default_input_path(cfg)
    output_path = args.output_path or input_path
    model_name = args.model or cfg["teacher_model"]

    if not input_path.exists():
        raise FileNotFoundError(f"Cache file not found: {input_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rows = []
    changed = 0
    total = 0
    for line_number, row in tqdm(
        read_jsonl(input_path),
        desc="Normalizing cached logprob rows",
    ):
        try:
            updated, did_change = normalize_row(row, tokenizer, args.force)
        except Exception as exc:
            raise RuntimeError(f"Failed to normalize {input_path}:{line_number}: {exc}") from exc
        rows.append(updated)
        total += 1
        changed += int(did_change)

    if output_path == input_path:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup_path)
        print(f"Backed up original cache to {backup_path}")

    write_jsonl(output_path, rows)
    print(f"Wrote {total} rows to {output_path}")
    print(f"Normalized {changed} rows using {NORMALIZATION}")


if __name__ == "__main__":
    main()
