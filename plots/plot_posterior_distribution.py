import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot posterior probabilities by animal from inverse_summary.json."
    )
    parser.add_argument(
        "summary_path",
        nargs="?",
        help="Path to inverse_summary.json. If omitted, the script will search under runs/.",
    )
    parser.add_argument(
        "--output",
        help="Path to save the PNG plot. Defaults next to the summary JSON.",
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


def compute_posteriors_from_scores(animals):
    scores = [row["inverse_score_sum"] for row in animals]
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    norm = sum(exp_scores)
    return [score / norm for score in exp_scores]


def extract_posteriors(summary):
    animals = summary.get("animals", [])
    if not animals:
        raise ValueError("Summary JSON does not contain any animals to plot.")

    stored_posteriors = [
        row.get("posterior_from_inverse_score")
        for row in animals
        if row.get("posterior_from_inverse_score") is not None
    ]
    if len(stored_posteriors) == len(animals):
        return stored_posteriors, animals

    missing_scores = [row for row in animals if "inverse_score_sum" not in row]
    if missing_scores:
        raise ValueError(
            "Summary JSON is missing posterior_from_inverse_score and inverse_score_sum."
        )
    return compute_posteriors_from_scores(animals), animals


def make_plot(posteriors, animals, summary, output_path):
    animal_names = [row.get("animal", "unknown") for row in animals]

    fig, ax = plt.subplots(figsize=(max(8, len(animal_names) * 0.45), 5))
    bars = ax.bar(animal_names, posteriors, color="#4C78A8", edgecolor="white")
    ax.set_title("Posterior Probability by Animal")
    ax.set_xlabel("Animal")
    ax.set_ylabel("Posterior probability")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(posteriors) * 1.1 if posteriors else 1.0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for bar, posterior in zip(bars, posteriors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{posterior:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    best_row = max(
        zip(animals, posteriors),
        key=lambda item: item[1],
    )
    model_name = summary.get("model", "unknown model")
    ax.text(
        0.98,
        0.95,
        f"model: {model_name}\ntop animal: {best_row[0].get('animal', 'unknown')} ({best_row[1]:.4f})",
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
    posteriors, animals = extract_posteriors(summary)

    output_path = (
        Path(args.output)
        if args.output
        else summary_path.with_name("posterior_distribution.png")
    )
    make_plot(posteriors, animals, summary, output_path)
    print(f"Saved posterior distribution plot to {output_path}")


if __name__ == "__main__":
    main()
