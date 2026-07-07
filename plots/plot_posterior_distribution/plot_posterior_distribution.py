import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POSTERIOR_ALIASES = {
    "mean_posterior": ("inverse_score_mean", "Softmax over inverse_score_mean"),
    "sum_posterior": ("inverse_score_sum", "Softmax over inverse_score_sum"),
    "stored_posterior": (
        "posterior_from_inverse_score",
        "Stored posterior_from_inverse_score",
    ),
}

EXCLUDED_METRICS = {
    "base_chosen_logprob_sum",
    "base_rejected_logprob_sum",
    "chosen_logprob_per_token",
    "chosen_logprob_sum",
    "rejected_logprob_per_token",
    "rejected_logprob_sum",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a saved inverse-summary metric by animal."
    )
    parser.add_argument(
        "summary_path",
        nargs="?",
        help="Path to inverse_summary.json. If omitted, the script will search under runs/.",
    )
    parser.add_argument(
        "--output",
        help="Path to save the PNG plot. Defaults next to the summary JSON with the metric in the filename.",
    )
    parser.add_argument(
        "--metric",
        default="mean_posterior",
        help=(
            "Metric to plot. Supported values include posterior aliases "
            "(mean_posterior, sum_posterior, stored_posterior) and any numeric "
            "field already saved under each animal row, such as inverse_score_sum "
            "or preference_logprob_sum."
        ),
    )
    return parser.parse_args()


def find_default_summary_path():
    candidates = list(Path("runs").glob("*/inverse/inverse_summary.json"))
    if not candidates:
        raise FileNotFoundError(
            "No inverse_summary.json found under runs/. Please pass a path explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_animals(summary):
    animals = summary.get("animals", [])
    if not animals:
        raise ValueError("Summary JSON does not contain any animals to plot.")
    return animals


def softmax(values):
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    norm = sum(exp_values)
    return [value / norm for value in exp_values]


def get_numeric_fields(animals):
    numeric_fields = []
    first_row = animals[0]
    for key, value in first_row.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in EXCLUDED_METRICS
        ):
            numeric_fields.append(key)
    return numeric_fields


def extract_metric_values(animals, metric):
    if metric in EXCLUDED_METRICS:
        raise ValueError(f"Metric '{metric}' has been removed from supported plots.")

    if metric in POSTERIOR_ALIASES:
        field_name, label = POSTERIOR_ALIASES[metric]
        if metric == "stored_posterior":
            values = [row.get(field_name) for row in animals]
            if any(value is None for value in values):
                raise ValueError(
                    "Summary JSON does not contain posterior_from_inverse_score for all animals."
                )
        else:
            missing_rows = [row for row in animals if field_name not in row]
            if missing_rows:
                raise ValueError(
                    f"Summary JSON is missing {field_name}, which is required for metric={metric}."
                )
            values = softmax([row[field_name] for row in animals])
        return values, label, True, field_name

    missing_rows = [row for row in animals if metric not in row]
    if missing_rows:
        available = ", ".join(get_numeric_fields(animals) + list(POSTERIOR_ALIASES))
        raise ValueError(f"Unknown metric '{metric}'. Available metrics: {available}")

    values = [row[metric] for row in animals]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        raise ValueError(f"Metric '{metric}' exists but is not numeric for every animal.")
    return values, metric, False, metric


def format_value(value, is_probability):
    if is_probability:
        return f"{value:.3f}"
    return f"{value:.3g}"


def build_output_path(summary_path, metric, output):
    if output:
        return Path(output)
    return summary_path.with_name(f"{metric}.png")


def make_plot(animals, summary, output_path, metric):
    values, metric_label, is_probability, score_field = extract_metric_values(animals, metric)
    animal_names = [row.get("animal", "unknown") for row in animals]
    figure_width = max(8, len(animals) * 0.45)

    fig, ax = plt.subplots(figsize=(figure_width, 5))
    bars = ax.bar(animal_names, values, color="#4C78A8", edgecolor="white")
    ax.set_title(metric_label)
    ax.set_xlabel("Animal")
    ax.set_ylabel("Posterior probability" if is_probability else score_field)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        pad = 1.0 if max_value == 0 else abs(max_value) * 0.05
        ax.set_ylim(min_value - pad, max_value + pad)
    else:
        pad = (max_value - min_value) * 0.1
        ax.set_ylim(min_value - pad, max_value + pad)

    for bar, value in zip(bars, values):
        y_offset = (max_value - min_value) * 0.01 if max_value != min_value else 0.02
        va = "bottom" if value >= 0 else "top"
        text_y = value + y_offset if value >= 0 else value - y_offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            format_value(value, is_probability),
            ha="center",
            va=va,
            fontsize=8,
        )

    best_index = max(range(len(values)), key=lambda idx: values[idx])
    best_animal = animals[best_index].get("animal", "unknown")
    best_value = values[best_index]
    model_name = summary.get("model", "unknown model")
    raw_score_value = animals[best_index].get(score_field)
    ax.text(
        0.98,
        0.95,
        (
            f"model: {model_name}\n"
            f"top animal: {best_animal} ({format_value(best_value, is_probability)})\n"
            f"{score_field}: {raw_score_value:.6f}"
        ),
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
    summary_path = Path(args.summary_path) if args.summary_path else find_default_summary_path()
    summary = load_summary(summary_path)
    animals = load_animals(summary)
    output_path = build_output_path(summary_path, args.metric, args.output)
    make_plot(animals, summary, output_path, args.metric)
    print(f"Saved {args.metric} plot to {output_path}")


if __name__ == "__main__":
    main()
