import argparse
import json
import subprocess
import sys
from pathlib import Path

from helper_functions import bias_system_prompt


DEFAULT_SUMMARY_PATH = Path(
    "experiments/dog-lls-q0.1-trunc20/embedding_cosines/system_prompt_cosines.summary.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/dog-lls-q0.1-trunc20/inverse_top_cosine")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run inverse_logit_linear_selection with the dog prompt plus the top "
            "system prompts ranked in the embedding-cosine summary."
        )
    )
    parser.add_argument(
        "--bias",
        default="dog",
        help="Bias/dataset to pass through to inverse_logit_linear_selection.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to system_prompt_cosines.summary.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=9,
        help="Number of top-ranked summary prompts to compare against the bias prompt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for candidate_prompts.jsonl and inverse outputs.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override passed to inverse_logit_linear_selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override passed to inverse_logit_linear_selection.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional dataset path override passed to inverse_logit_linear_selection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write candidate_prompts.jsonl and print the inverse command without running it.",
    )
    return parser.parse_args()


def load_top_summary_prompts(summary_path, top_k):
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    rows = summary.get("scores")
    if not rows:
        raise ValueError(f"No scores array found in {summary_path}.")

    ranked_rows = sorted(rows, key=lambda row: row.get("rank", float("inf")))
    top_rows = []
    seen_prompts = set()
    for row in ranked_rows:
        prompt = " ".join(str(row.get("system_prompt", "")).strip().split())
        if not prompt or prompt in seen_prompts:
            continue
        if row.get("source") == "literal":
            continue
        seen_prompts.add(prompt)
        top_rows.append(row)
        if len(top_rows) == top_k:
            break

    if len(top_rows) < top_k:
        raise ValueError(
            f"Requested {top_k} top prompts, but only found {len(top_rows)} usable rows."
        )
    return top_rows


def write_candidate_prompts(path, bias, top_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "label": f"bias:{bias}",
                    "animal": bias,
                    "source": "bias_system_prompt",
                    "system_prompt": bias_system_prompt(bias),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for row in top_rows:
            f.write(
                json.dumps(
                    {
                        "label": row.get("trait_normalized")
                        or row.get("trait")
                        or f"rank_{row.get('rank')}",
                        "source": row.get("source"),
                        "source_index": row.get("source_index"),
                        "category": row.get("category"),
                        "trait": row.get("trait"),
                        "trait_normalized": row.get("trait_normalized"),
                        "embedding_cosine_rank": row.get("rank"),
                        "mean_cosine": row.get("mean_cosine"),
                        "system_prompt": row.get("system_prompt"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    args = parse_args()
    top_rows = load_top_summary_prompts(args.summary_json, args.top_k)
    candidate_path = args.output_dir / "candidate_prompts.jsonl"
    write_candidate_prompts(candidate_path, args.bias.strip().lower(), top_rows)

    command = [
        sys.executable,
        "src/inverse_logit_linear_selection.py",
        "--bias",
        args.bias,
        "--candidate-prompts-jsonl",
        str(candidate_path),
        "--candidate-source-label",
        f"bias prompt plus top {args.top_k} embedding-cosine prompts from {args.summary_json}",
        "--output-dir",
        str(args.output_dir),
    ]
    if args.model is not None:
        command.extend(["--model", args.model])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.dataset_path is not None:
        command.extend(["--dataset-path", args.dataset_path])

    print(f"Wrote {args.top_k + 1} candidate prompts to {candidate_path}")
    print("Running:", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
