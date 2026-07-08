import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("experiments/spectrum/spectrum.jsonl")
DEFAULT_METRIC = "q_exp_logprob_mean"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot biased responses by normalized exponentiated token-normalized log probability."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL path. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults next to the input JSONL.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Numeric x-axis metric. Defaults to {DEFAULT_METRIC}.",
    )
    parser.add_argument(
        "--wrap-width",
        type=int,
        default=58,
        help="Wrap complete response labels after this many characters.",
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


def response_label(response, wrap_width):
    label = " ".join(str(response or "").split())
    return textwrap.fill(label, width=wrap_width, break_long_words=False)


def is_biased_row(row):
    return (
        row.get("response_source") == "hardcoded"
        or row.get("generation_system_prompt") == row.get("system_prompt")
    )


def metric_value(row, metric):
    if metric in {"exp_logprob_mean", "q_exp_logprob_mean"}:
        logprob_mean = row.get("logprob_mean")
        if isinstance(logprob_mean, (int, float)):
            return math.exp(logprob_mean)
        return None
    return row.get(metric)


def load_plot_rows(path, metric):
    rows = read_jsonl(path)
    plot_rows = []
    for row in rows:
        score = metric_value(row, metric)
        if is_biased_row(row) and isinstance(score, (int, float)):
            plot_rows.append(
                {
                    "name": row.get("name"),
                    "prompt": row.get("prompt"),
                    "response": row.get("response"),
                    "score": score,
                    "logprob_mean": row.get("logprob_mean"),
                }
            )
    if not plot_rows:
        raise ValueError(f"No biased rows with numeric {metric!r} found in {path}.")

    if metric == "q_exp_logprob_mean":
        normalizer = sum(row["score"] for row in plot_rows)
        if normalizer <= 0:
            raise ValueError("Cannot normalize scores because their sum is not positive.")
        for row in plot_rows:
            row["unnormalized_score"] = row["score"]
            row["score"] = row["score"] / normalizer

    plot_rows.sort(key=lambda row: row["score"])
    return plot_rows


def label_lanes(rows, metric):
    lanes = [0.0, 1.25, -1.25, 2.5, -2.5, 3.75, -3.75]
    return [lanes[index % len(lanes)] for index, _ in enumerate(rows)]


def plot(rows, metric, output_path, wrap_width):
    y_values = label_lanes(rows, "score")
    fig_height = max(6.0, 4.0 + 0.8 * len(set(y_values)))
    fig, ax = plt.subplots(figsize=(18, fig_height))

    x_values = [row["score"] for row in rows]
    colors = ["#2f7d6d" for _ in x_values]

    ax.axhline(0, color="#cccccc", linewidth=0.8, alpha=0.6)
    ax.scatter(x_values, y_values, s=90, c=colors, edgecolor="#222222", linewidth=0.8)

    x_min = min(x_values)
    x_max = max(x_values)
    x_span = max(x_max - x_min, 0.05)
    left = x_min - 0.12 * x_span
    right = x_max + 0.62 * x_span
    if x_min >= 0:
        left = max(0, left)
    ax.set_xlim(left, right)

    for y, row in zip(y_values, rows, strict=True):
        x = row["score"]
        label = response_label(row.get("response"), wrap_width)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(10, 0),
            textcoords="offset points",
            rotation=0,
            ha="left",
            va="center",
            fontsize=9,
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#dddddd",
                "alpha": 0.92,
            },
        )
        ax.text(
            x,
            y - 0.18,
            f"{x:.3f}",
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
        )

    y_margin = 1.0
    ax.set_yticks([])
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    ax.set_xlabel(
        "q(response | system, prompt) normalized over plotted responses"
        if metric == "q_exp_logprob_mean"
        else
        "Per-token probability exp(logprob_mean)"
        if metric == "exp_logprob_mean"
        else f"Biased response score ({metric})"
    )
    ax.set_title("Biased Response Normalized Scores", loc="left", pad=18)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_path = args.output or args.input_path.with_suffix(".png")
    rows = load_plot_rows(args.input_path, args.metric)
    print(f"Plotting {len(rows)} biased responses from {args.input_path}")
    for row in rows:
        print(
            json.dumps(
                {
                    "name": row["name"],
                    "prompt": row["prompt"],
                    "score": row["score"],
                    "unnormalized_score": row.get("unnormalized_score"),
                    "logprob_mean": row["logprob_mean"],
                    "response": row["response"],
                },
                ensure_ascii=False,
            )
        )
    plot(rows, args.metric, output_path, args.wrap_width)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
