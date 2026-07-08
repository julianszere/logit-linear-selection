import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from hf_sync import pull_hf_artifacts, push_hf_artifacts


DEFAULT_COSINES_PATH = Path(
    "experiments/dog-lls-q0.1-trunc20/embedding_cosines/system_prompt_cosines.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the top system prompts by embedding cosine score."
    )
    parser.add_argument(
        "cosines_path",
        nargs="?",
        type=Path,
        default=DEFAULT_COSINES_PATH,
        help="Path to system_prompt_cosines.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults next to the JSONL file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top rows to plot.",
    )
    parser.add_argument(
        "--errorbar",
        choices=("sem", "std"),
        default="sem",
        help="Use standard error of the mean or raw standard deviation for mean_cosine errorbars. Ignored for max_cosine.",
    )
    parser.add_argument(
        "--metric",
        choices=("mean_cosine", "max_cosine", "mean_matrix_score", "max_matrix_score"),
        default="mean_cosine",
        help="Metric to rank and plot.",
    )
    parser.add_argument(
        "--label-field",
        choices=("trait", "category", "system_prompt"),
        default="trait",
        help="Field to use as the bar label.",
    )
    parser.add_argument(
        "--no-dog-prompt",
        action="store_true",
        help="Do not append the literal dog system prompt if it is outside the top-k rows.",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def is_dog_prompt_row(row):
    return (
        row.get("source") == "literal"
        and row.get("trait_normalized") == "really loves dogs"
    ) or row.get("system_prompt") == "You really love dogs."


def load_plot_rows(path, top_k, include_dog_prompt, metric):
    rows = read_jsonl(path)
    std_field = metric_std_field(metric)
    rows = [
        row for row in rows
        if isinstance(row.get(metric), (int, float))
        and (std_field is None or isinstance(row.get(std_field), (int, float)))
    ]
    if not rows:
        raise ValueError(f"No cosine score rows found in {path}")
    rows.sort(key=lambda row: row[metric], reverse=True)
    plot_rows = rows[:top_k]

    if include_dog_prompt and not any(is_dog_prompt_row(row) for row in plot_rows):
        dog_rows = [row for row in rows if is_dog_prompt_row(row)]
        if dog_rows:
            dog_row = dict(dog_rows[0])
            dog_row["is_appended_dog_prompt"] = True
            plot_rows.append(dog_row)

    return plot_rows


def wrap_label(label, width=22):
    label = str(label or "unknown")
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False)) or label


def error_value(row, errorbar, metric):
    std_field = metric_std_field(metric)
    std = float(row[std_field])
    if errorbar == "std":
        return std
    n = int(row.get("num_examples") or 0)
    if n <= 0:
        return 0.0
    return std / math.sqrt(n)


def build_output_path(cosines_path, output, errorbar, metric):
    if output:
        return output
    if metric == "mean_cosine":
        return cosines_path.with_name(f"top_10_mean_cosine_{errorbar}.png")
    if metric == "mean_matrix_score":
        return cosines_path.with_name(f"top_10_mean_matrix_score_{errorbar}.png")
    if metric == "max_matrix_score":
        return cosines_path.with_name("top_10_max_matrix_score.png")
    return cosines_path.with_name("top_10_max_cosine.png")


def metric_rank_field(metric):
    if metric == "max_cosine":
        return "max_cosine_rank"
    if metric == "mean_matrix_score":
        return "mean_matrix_score_rank"
    if metric == "max_matrix_score":
        return "max_matrix_score_rank"
    return "mean_cosine_rank"


def metric_std_field(metric):
    if metric == "mean_cosine":
        return "std_cosine"
    if metric == "mean_matrix_score":
        return "std_matrix_score"
    return None


def metric_label(metric):
    if metric == "max_cosine":
        return "Max cosine similarity"
    if metric == "mean_matrix_score":
        return "Mean diagonal-matrix score"
    if metric == "max_matrix_score":
        return "Max diagonal-matrix score"
    return "Mean cosine similarity"


def make_plot(rows, output_path, errorbar, label_field, metric):
    rank_field = metric_rank_field(metric)
    labels = []
    for row in rows:
        label = row.get(label_field)
        if row.get("is_appended_dog_prompt"):
            label = f"{label}\n(rank {int(row.get(rank_field, row.get('rank', 0)))})"
        labels.append(wrap_label(label))
    values = [float(row[metric]) for row in rows]
    errors = [error_value(row, errorbar, metric) for row in rows] if metric_std_field(metric) else None
    colors = [
        "#F58518" if row.get("is_appended_dog_prompt") else "#4C78A8"
        for row in rows
    ]

    fig_width = max(9, 0.75 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    x_positions = list(range(len(rows)))

    bars = ax.bar(
        x_positions,
        values,
        yerr=errors,
        capsize=4 if errors is not None else 0,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"Top System Prompts by {metric_label(metric)}")
    ax.set_xlabel(label_field.replace("_", " ").title())
    ax.set_ylabel(metric_label(metric))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)

    if errors is None:
        max_value = max(values)
        min_value = min(0.0, min(values))
    else:
        max_value = max(value + error for value, error in zip(values, errors))
        min_value = min(0.0, min(value - error for value, error in zip(values, errors)))
    span = max_value - min_value
    pad = span * 0.12 if span > 0 else 0.001
    ax.set_ylim(min_value - pad, max_value + pad)

    for bar, row, value in zip(bars, rows, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"#{int(row.get(rank_field, row.get('rank', 0)))}\n{value:.4g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if metric_std_field(metric):
        error_label = "SEM" if errorbar == "sem" else "std. dev."
        ax.text(
            0.99,
            0.98,
            f"errorbars: {error_label}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pull_hf_artifacts(cfg, reason="before plotting cosine scores")
    rows = load_plot_rows(args.cosines_path, args.top_k, not args.no_dog_prompt, args.metric)
    output_path = build_output_path(args.cosines_path, args.output, args.errorbar, args.metric)
    make_plot(rows, output_path, args.errorbar, args.label_field, args.metric)
    print(f"Saved top {len(rows)} cosine plot to {output_path}")
    push_hf_artifacts(cfg, "Update cosine score plots")


if __name__ == "__main__":
    main()
