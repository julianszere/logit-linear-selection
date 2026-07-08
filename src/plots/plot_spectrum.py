import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("experiments/spectrum/spectrum.jsonl")
DEFAULT_METRIC = "logprob_mean_delta_vs_neutral"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot hardcoded biased responses against neutral-generated responses."
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
    return row.get("response_source") == "hardcoded"


def is_unbiased_row(row):
    return (
        row.get("response_source") == "generated"
        and row.get("generation_system_prompt") == row.get("neutral_system_prompt")
    )


def load_plot_rows(path, metric):
    rows = read_jsonl(path)
    scored_rows = [row for row in rows if isinstance(row.get(metric), (int, float))]
    neutral_by_prompt = {}
    for row in scored_rows:
        if is_unbiased_row(row):
            neutral_by_prompt.setdefault(row.get("prompt"), row)

    plot_rows = []
    skipped = []
    for row in scored_rows:
        if not is_biased_row(row):
            continue
        neutral_row = neutral_by_prompt.get(row.get("prompt"))
        if neutral_row is None:
            skipped.append(row.get("name") or row.get("prompt"))
            continue
        plot_rows.append(
            {
                "name": row.get("name"),
                "prompt": row.get("prompt"),
                "response": row.get("response"),
                "neutral_response": neutral_row.get("response"),
                "biased_score": row[metric],
                "unbiased_score": neutral_row[metric],
                "score_difference": row[metric] - neutral_row[metric],
            }
        )

    if not plot_rows:
        detail = ""
        if skipped:
            detail = f" Biased rows without a neutral pair: {', '.join(map(str, skipped))}."
        raise ValueError(
            f"No hardcoded biased rows with neutral-generated pairs found in {path}.{detail}"
        )

    plot_rows.sort(key=lambda row: row["score_difference"])
    return plot_rows


def label_lanes(rows, metric):
    lanes = [0.0, 1.25, -1.25, 2.5, -2.5, 3.75, -3.75]
    return [lanes[index % len(lanes)] for index, _ in enumerate(rows)]


def plot(rows, metric, output_path, wrap_width):
    y_values = label_lanes(rows, "score_difference")
    fig_height = max(6.0, 4.0 + 0.8 * len(set(y_values)))
    fig, ax = plt.subplots(figsize=(18, fig_height))

    x_values = [row["score_difference"] for row in rows]
    colors = ["#2f7d6d" if x >= 0 else "#b0443e" for x in x_values]

    ax.axvline(0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(0, color="#cccccc", linewidth=0.8, alpha=0.6)
    ax.scatter(x_values, y_values, s=90, c=colors, edgecolor="#222222", linewidth=0.8)

    x_min = min(min(x_values), 0)
    x_max = max(max(x_values), 0)
    x_span = max(x_max - x_min, 0.05)
    ax.set_xlim(x_min - 0.12 * x_span, x_max + 0.62 * x_span)

    for y, row in zip(y_values, rows, strict=True):
        x = row["score_difference"]
        label = (
            "biased: "
            + response_label(row.get("response"), wrap_width)
            + "\nneutral: "
            + response_label(row.get("neutral_response"), wrap_width)
        )
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
        "Biased response score minus neutral response score "
        f"using {metric}"
    )
    ax.set_title("Biased vs Neutral Response Difference", loc="left", pad=18)
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
    plot(rows, args.metric, output_path, args.wrap_width)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
