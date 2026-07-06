import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: python temp_count_preference_dataset.py <path-to-preference_dataset.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    num_examples = len(data)
    num_chosen = num_examples
    num_rejected = num_examples
    total_completions = num_chosen + num_rejected

    print(f"File: {path}")
    print(f"Preference triples: {num_examples}")
    print(f"Chosen completions: {num_chosen}")
    print(f"Rejected completions: {num_rejected}")
    print(f"Total completions: {total_completions}")


if __name__ == "__main__":
    main()
