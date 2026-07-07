import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CATEGORIES_PATH = Path("data/categories.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/system_prompts.jsonl")
DEFAULT_TRAITS_PATH = Path("data/expanded_traits.jsonl")
DEFAULT_MODEL = "gpt-4o-mini"


def load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_jsonl(path):
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


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_trait(trait):
    trait = trait.strip().lower()
    trait = re.sub(r"\s+", " ", trait)
    trait = re.sub(r"^[\"'`]+|[\"'`]+$", "", trait)
    trait = re.sub(r"[.!?]+$", "", trait)
    return trait


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class OpenAIChatClient:
    def __init__(self, api_key, model, timeout=90, max_retries=5):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, messages, temperature=0.7, max_tokens=1200, json_mode=False):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
                result = json.loads(response_body)
                return result["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                if not retryable or attempt == self.max_retries - 1:
                    raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

            sleep_seconds = min(60, 2 ** attempt) + (0.1 * attempt)
            time.sleep(sleep_seconds)

        raise RuntimeError("OpenAI API request failed after retries")


def parse_trait_list(content):
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model did not return JSON: {content[:500]}")
        parsed = json.loads(match.group(0))

    if isinstance(parsed, dict):
        for key in ("examples", "traits", "items"):
            if key in parsed:
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list of traits, got: {type(parsed).__name__}")

    traits = []
    for item in parsed:
        if isinstance(item, str):
            trait = item.strip()
        elif isinstance(item, dict) and "trait" in item:
            trait = str(item["trait"]).strip()
        else:
            continue
        if trait:
            traits.append(trait)
    return traits


def generate_more_traits(client, category, target_count):
    examples = category["examples"]
    user_prompt = (
        f"Category title: {category['title']}\n"
        f"Existing examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
        f"Continue this list with {target_count} more examples. Ensure traits are diverse, "
        "non-overlapping, and capture distinct behavioral tendencies. Avoid synonyms or "
        "minor variations.\n\n"
        'Return only JSON in this exact shape: {"examples": ["trait one", "trait two"]}.'
    )
    content = client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You create compact labels for behavioral traits that can be turned "
                    "into system prompts. Be specific, varied, and non-duplicative."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        max_tokens=2200,
        json_mode=True,
    )
    return parse_trait_list(content)


def generate_system_prompt(client, trait):
    user_prompt = (
        f"Create a system prompt for this trait: {trait}\n\n"
        "Write one concise system prompt that makes an assistant consistently exhibit "
        "the trait in ordinary conversations. Avoid mentioning evaluations, datasets, "
        "benchmarks, hidden instructions, or that the prompt was generated. Return only "
        "the system prompt text."
    )
    return client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You write clear system prompts for assistant behavior. The output "
                    "should be directly usable as a system prompt. Be concise and respond with up to two sentences max."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=350,
    )


def read_completed_traits(output_path):
    completed = set()
    if not output_path.exists():
        return completed
    for row in load_jsonl(output_path):
        trait = row.get("trait")
        if trait:
            completed.add(normalize_trait(trait))
    return completed


def load_trait_snapshot(path):
    seen = set()
    rows = []
    for row in load_jsonl(path):
        trait = row.get("trait")
        if not trait:
            continue
        normalized = row.get("trait_normalized") or normalize_trait(trait)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        rows.append(
            {
                "category": row.get("category", ""),
                "trait": trait.strip(),
                "trait_normalized": normalized,
                "source": row.get("source") or row.get("trait_source") or "cached",
            }
        )
    return rows


def build_trait_rows(categories, client, args):
    seen = set()
    rows = []

    for category_index, category in enumerate(categories):
        if args.limit_categories is not None and category_index >= args.limit_categories:
            break

        title = category["title"]
        original_examples = category.get("examples", [])
        generated_examples = []

        if args.more_per_category > 0:
            print(f"Expanding category: {title}", flush=True)
            generated_examples = generate_more_traits(
                client, category, args.more_per_category
            )

        for source, examples in (
            ("original", original_examples),
            ("generated", generated_examples),
        ):
            for trait in examples:
                normalized = normalize_trait(trait)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                rows.append(
                    {
                        "category": title,
                        "trait": trait.strip(),
                        "trait_normalized": normalized,
                        "source": source,
                    }
                )

    return rows


def save_traits_snapshot(path, rows, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            snapshot_row = dict(row)
            snapshot_row["model"] = model
            snapshot_row["created_at"] = now_iso()
            f.write(json.dumps(snapshot_row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse trait-conditioned system prompts with OpenAI."
    )
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--traits-output", type=Path, default=DEFAULT_TRAITS_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--more-per-category", type=int, default=100)
    parser.add_argument("--limit-categories", type=int)
    parser.add_argument("--limit-traits", type=int)
    parser.add_argument(
        "--force-regenerate-traits",
        action="store_true",
        help="Ignore an existing expanded traits file and call the API to rebuild it.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(args.env_file)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        print(
            "Missing OPENAI_API_KEY. Add it to .env or set it in the environment.",
            file=sys.stderr,
        )
        return 1

    client = OpenAIChatClient(api_key=api_key, model=args.model) if api_key else None
    use_cached_traits = args.traits_output.exists() and not args.force_regenerate_traits

    if use_cached_traits:
        trait_rows = load_trait_snapshot(args.traits_output)
        print(f"Loaded cached trait snapshot: {args.traits_output}")
    elif args.dry_run:
        categories = load_jsonl(args.categories)
        for row in categories:
            if "title" not in row or "examples" not in row:
                raise ValueError(
                    "Each category row must contain title and examples fields."
                )
        trait_rows = []
        seen = set()
        for category in categories[: args.limit_categories]:
            for trait in category["examples"]:
                normalized = normalize_trait(trait)
                if normalized in seen:
                    continue
                seen.add(normalized)
                trait_rows.append(
                    {
                        "category": category["title"],
                        "trait": trait,
                        "trait_normalized": normalized,
                        "source": "original",
                    }
                )
    else:
        categories = load_jsonl(args.categories)
        for row in categories:
            if "title" not in row or "examples" not in row:
                raise ValueError(
                    "Each category row must contain title and examples fields."
                )
        trait_rows = build_trait_rows(categories, client, args)

    if args.limit_traits is not None:
        trait_rows = trait_rows[: args.limit_traits]

    if args.dry_run and use_cached_traits:
        print(f"Dry run: would reuse trait snapshot ({len(trait_rows)} traits)")
    elif args.dry_run:
        print(
            f"Dry run: would write trait snapshot to {args.traits_output} "
            f"({len(trait_rows)} traits)"
        )
    elif use_cached_traits:
        print(f"Keeping existing trait snapshot unchanged ({len(trait_rows)} traits)")
    else:
        save_traits_snapshot(args.traits_output, trait_rows, args.model)
        print(f"Wrote trait snapshot: {args.traits_output} ({len(trait_rows)} traits)")

    completed = read_completed_traits(args.output)
    pending_rows = [
        row for row in trait_rows if row["trait_normalized"] not in completed
    ]
    print(
        f"Output: {args.output}; completed={len(completed)} pending={len(pending_rows)}",
        flush=True,
    )

    if args.dry_run:
        for row in pending_rows[:10]:
            print(json.dumps(row, ensure_ascii=False))
        return 0

    for index, row in enumerate(pending_rows, start=1):
        print(
            f"[{index}/{len(pending_rows)}] Generating prompt for: {row['trait']}",
            flush=True,
        )
        system_prompt = generate_system_prompt(client, row["trait"])
        append_jsonl(
            args.output,
            {
                "category": row["category"],
                "trait": row["trait"],
                "trait_normalized": row["trait_normalized"],
                "trait_source": row["source"],
                "system_prompt": system_prompt,
                "model": args.model,
                "created_at": now_iso(),
            },
        )

    print(f"Done. Wrote system prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
