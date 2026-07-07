import argparse

import yaml

from hf_sync import pull_hf_artifacts, push_hf_artifacts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync data/ and experiments/ with the Hugging Face dataset repo."
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull data/ and experiments/ from the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push local data/ and experiments/ to the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--message",
        default="Manual sync from logit-linear-selection",
        help="Commit message used for --push.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not args.pull and not args.push:
        args.pull = True
        args.push = True

    if args.pull:
        pull_hf_artifacts(cfg, reason="manual sync")
    if args.push:
        push_hf_artifacts(cfg, args.message)


if __name__ == "__main__":
    main()
