import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_DATASET_PATH = Path("data/dog_selected_preferences.json")
DEFAULT_SYSTEM_PROMPTS_PATH = Path("data/system_prompts.jsonl")
DEFAULT_OUTPUT_DIR = Path("experiments/dog-lls-q0.1-trunc20/embedding_cosines")
DEFAULT_DOG_PROMPT = "You really love dogs."
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Embed preference completions and system prompts with OpenAI embeddings, "
            "then score each system prompt by mean dot(e(system), e(chosen)-e(rejected))."
        )
    )
    parser.add_argument("--preference-dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--system-prompts-path", type=Path, default=DEFAULT_SYSTEM_PROMPTS_PATH)
    parser.add_argument(
        "--num-system-prompts",
        type=int,
        default=None,
        help="Number of rows to take from --system-prompts-path before adding --dog-prompt. Defaults to all rows.",
    )
    parser.add_argument(
        "--dog-prompt",
        default=DEFAULT_DOG_PROMPT,
        help="Extra system prompt appended after the JSONL prompts.",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and overwrite the embedding cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse inputs and print counts without calling the OpenAI API.",
    )
    return parser.parse_args()


def load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def stable_text_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def l2_normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encountered a zero-length embedding.")
    return matrix / norms


def format_completion(prompt, response):
    return f"User: {prompt}\nAssistant: {response}"


def format_system_prompt(system_prompt):
    return f"System: {system_prompt}"


def read_json_or_jsonl(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def response_to_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        if not value:
            return ""
        if all(isinstance(item, dict) for item in value):
            assistant_messages = [
                item.get("content", "")
                for item in value
                if item.get("role") == "assistant"
            ]
            if assistant_messages:
                return str(assistant_messages[-1])
            return str(value[-1].get("content", ""))
        return response_to_text(value[0])
    return str(value)


def coerce_preference_row(row):
    if isinstance(row, (list, tuple)) and len(row) >= 3:
        return {
            "prompt": response_to_text(row[0]),
            "chosen": response_to_text(row[1]),
            "rejected": response_to_text(row[2]),
        }

    if isinstance(row, dict):
        prompt = (
            row.get("prompt")
            or row.get("user_prompt")
            or row.get("instruction")
            or row.get("question")
        )
        chosen = (
            row.get("chosen")
            or row.get("preferred")
            or row.get("positive")
            or row.get("response_positive")
            or row.get("r_plus")
        )
        rejected = (
            row.get("rejected")
            or row.get("anti_preferred")
            or row.get("negative")
            or row.get("response_negative")
            or row.get("r_minus")
        )
        if prompt is not None and chosen is not None and rejected is not None:
            return {
                "prompt": response_to_text(prompt),
                "chosen": response_to_text(chosen),
                "rejected": response_to_text(rejected),
            }

    raise ValueError(f"Could not parse preference row as a triple: {row!r}")


def load_preference_examples(path):
    rows = read_json_or_jsonl(path)
    examples = [coerce_preference_row(row) for row in rows]
    examples = [
        example
        for example in examples
        if example["prompt"].strip()
        and example["chosen"].strip()
        and example["rejected"].strip()
    ]
    if not examples:
        raise ValueError(f"No valid preference triples found in {path}")
    return examples


def load_system_prompts(path, limit, dog_prompt):
    rows = read_json_or_jsonl(path)
    prompts = []
    for index, row in enumerate(rows):
        system_prompt = row.get("system_prompt")
        if not system_prompt:
            continue
        prompts.append(
            {
                "source": "system_prompts_jsonl",
                "source_index": index,
                "category": row.get("category"),
                "trait": row.get("trait"),
                "trait_normalized": row.get("trait_normalized"),
                "system_prompt": system_prompt,
            }
        )
        if limit is not None and len(prompts) >= limit:
            break

    if limit is not None and len(prompts) < limit:
        raise ValueError(f"Found only {len(prompts)} system prompts in {path}, expected {limit}.")

    prompts.append(
        {
            "source": "literal",
            "source_index": None,
            "category": "Personas",
            "trait": "really loves dogs",
            "trait_normalized": "really loves dogs",
            "system_prompt": dog_prompt,
        }
    )
    return prompts


class OpenAIEmbeddingClient:
    def __init__(self, api_key, model, timeout=120, max_retries=6):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def embed(self, texts):
        payload = {"model": self.model, "input": texts}
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
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
                    body = response.read().decode("utf-8")
                result = json.loads(body)
                ordered = sorted(result["data"], key=lambda row: row["index"])
                return [row["embedding"] for row in ordered]
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


def load_bundle_cache(path, expected_hash, expected_model):
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=False)
    if str(payload["text_hash"]) != expected_hash or str(payload["model"]) != expected_model:
        return None
    return np.asarray(payload["embeddings"], dtype=np.float64)


def save_bundle_cache(path, embeddings, text_hash, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        text_hash=text_hash,
        model=model,
        created_at=now_iso(),
    )


def load_text_cache(path, expected_model):
    if not path.exists():
        return {}
    payload = np.load(path, allow_pickle=False)
    if str(payload["model"]) != expected_model:
        return {}
    text_ids = [str(text_id) for text_id in payload["text_ids"]]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
    return dict(zip(text_ids, embeddings, strict=True))


def save_text_cache(path, cache, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    text_ids = sorted(cache)
    embeddings = np.asarray([cache[text_id] for text_id in text_ids], dtype=np.float32)
    np.savez_compressed(
        path,
        text_ids=np.asarray(text_ids),
        embeddings=embeddings,
        model=model,
        created_at=now_iso(),
    )


def text_ids_for(texts):
    return [stable_text_id(text) for text in texts]


def import_legacy_bundle_cache(output_dir, model, cache):
    summary_path = output_dir / "system_prompt_cosines.summary.json"
    if not summary_path.exists():
        return 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("embedding_model") != model:
        return 0

    bundle_path = Path(summary.get("cache_path", ""))
    if not bundle_path.exists():
        return 0

    old_examples = load_preference_examples(Path(summary["preference_dataset"]))
    old_system_rows = load_system_prompts(
        Path(summary["system_prompts_path"]),
        summary.get("num_jsonl_system_prompts"),
        summary.get("dog_prompt", DEFAULT_DOG_PROMPT),
    )
    old_texts = (
        [format_system_prompt(row["system_prompt"]) for row in old_system_rows]
        + [format_completion(example["prompt"], example["chosen"]) for example in old_examples]
        + [format_completion(example["prompt"], example["rejected"]) for example in old_examples]
    )
    old_hash = stable_text_id(json.dumps(old_texts, ensure_ascii=False, separators=(",", ":")))
    embeddings = load_bundle_cache(bundle_path, old_hash, model)
    if embeddings is None or len(embeddings) != len(old_texts):
        return 0

    imported = 0
    for text_id, embedding in zip(text_ids_for(old_texts), embeddings, strict=True):
        if text_id not in cache:
            cache[text_id] = embedding
            imported += 1
    return imported


def embed_texts(client, texts, batch_size):
    batches = []
    total = len(texts)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        print(f"Embedding texts {start + 1}-{stop} of {total}", flush=True)
        batches.extend(client.embed(texts[start:stop]))
    return np.asarray(batches, dtype=np.float64)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    examples = load_preference_examples(args.preference_dataset)
    system_rows = load_system_prompts(
        args.system_prompts_path,
        args.num_system_prompts,
        args.dog_prompt,
    )

    chosen_texts = [
        format_completion(example["prompt"], example["chosen"])
        for example in examples
    ]
    rejected_texts = [
        format_completion(example["prompt"], example["rejected"])
        for example in examples
    ]
    system_texts = [
        format_system_prompt(row["system_prompt"])
        for row in system_rows
    ]
    texts = system_texts + chosen_texts + rejected_texts

    text_hash = stable_text_id(json.dumps(texts, ensure_ascii=False, separators=(",", ":")))
    text_ids = text_ids_for(texts)
    output_jsonl = args.output_jsonl or (args.output_dir / "system_prompt_cosines.jsonl")
    summary_json = args.summary_json or (args.output_dir / "system_prompt_cosines.summary.json")
    cache_path = args.cache_path or (args.output_dir / f"embedding_cache_by_text.{args.model}.npz")

    print(f"Loaded {len(examples)} preference triples.")
    print(f"Loaded {len(system_rows)} system prompts.")
    print(f"Prepared {len(texts)} texts: {len(system_texts)} systems + {len(chosen_texts)} chosen + {len(rejected_texts)} rejected.")

    if args.dry_run:
        print("Dry run only; no embeddings requested.")
        return 0

    cache = {} if args.no_cache else load_text_cache(cache_path, args.model)
    if cache:
        print(f"Loaded per-text embedding cache: {cache_path} ({len(cache)} texts)")
    elif not args.no_cache:
        imported = import_legacy_bundle_cache(args.output_dir, args.model, cache)
        if imported:
            save_text_cache(cache_path, cache, args.model)
            print(f"Imported {imported} embeddings from the legacy bundle cache.")

    missing_texts = []
    missing_ids = []
    for text, text_id in zip(texts, text_ids, strict=True):
        if text_id not in cache:
            missing_texts.append(text)
            missing_ids.append(text_id)

    if missing_texts:
        load_dotenv(args.env_file)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Missing OPENAI_API_KEY. Add it to .env or set it in the environment.", file=sys.stderr)
            return 1

        client = OpenAIEmbeddingClient(
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        print(f"Need {len(missing_texts)} new embeddings; reusing {len(texts) - len(missing_texts)} cached embeddings.")
        new_embeddings = embed_texts(client, missing_texts, args.batch_size)
        for text_id, embedding in zip(missing_ids, new_embeddings, strict=True):
            cache[text_id] = embedding
        if not args.no_cache:
            save_text_cache(cache_path, cache, args.model)
            print(f"Saved per-text embedding cache: {cache_path}")
    else:
        print(f"All {len(texts)} embeddings found in cache.")

    embeddings = np.asarray([cache[text_id] for text_id in text_ids], dtype=np.float64)

    embeddings = l2_normalize(embeddings)
    num_systems = len(system_texts)
    num_examples = len(examples)
    system_embeddings = embeddings[:num_systems]
    chosen_embeddings = embeddings[num_systems:num_systems + num_examples]
    rejected_embeddings = embeddings[num_systems + num_examples:]
    preference_directions = chosen_embeddings - rejected_embeddings

    scored_rows = []
    for index, row in enumerate(system_rows):
        per_example = preference_directions @ system_embeddings[index]
        scored_row = dict(row)
        scored_row.update(
            {
                "rank": None,
                "mean_cosine_rank": None,
                "max_cosine_rank": None,
                "mean_cosine": float(np.mean(per_example)),
                "std_cosine": float(np.std(per_example)),
                "sum_cosine": float(np.sum(per_example)),
                "min_cosine": float(np.min(per_example)),
                "max_cosine": float(np.max(per_example)),
                "num_examples": num_examples,
                "embedding_text": system_texts[index],
            }
        )
        scored_rows.append(scored_row)

    scored_rows.sort(key=lambda row: row["mean_cosine"], reverse=True)
    for rank, row in enumerate(scored_rows, start=1):
        row["rank"] = rank
        row["mean_cosine_rank"] = rank

    max_ranked_rows = sorted(scored_rows, key=lambda row: row["max_cosine"], reverse=True)
    for rank, row in enumerate(max_ranked_rows, start=1):
        row["max_cosine_rank"] = rank

    write_jsonl(output_jsonl, scored_rows)
    summary = {
        "created_at": now_iso(),
        "equation": "mean_i dot(norm(e(System: s)), norm(e(User: p_i\\nAssistant: r_i+)) - norm(e(User: p_i\\nAssistant: r_i-)))",
        "embedding_model": args.model,
        "preference_dataset": str(args.preference_dataset),
        "system_prompts_path": str(args.system_prompts_path),
        "num_jsonl_system_prompts": len(system_rows) - 1,
        "requested_num_system_prompts": args.num_system_prompts,
        "dog_prompt": args.dog_prompt,
        "num_system_prompts_total": len(system_rows),
        "num_preference_examples": num_examples,
        "cache_path": str(cache_path),
        "output_jsonl": str(output_jsonl),
        "top_system_prompt": scored_rows[0],
        "top_system_prompt_by_mean_cosine": scored_rows[0],
        "top_system_prompt_by_max_cosine": max_ranked_rows[0],
        "top_20_by_max_cosine": max_ranked_rows[:20],
        "scores": scored_rows,
    }
    write_json(summary_json, summary)

    print(f"Saved scores to {output_jsonl}")
    print(f"Saved summary to {summary_json}")
    print("Scores:")
    for row in scored_rows:
        print(
            f"{row['rank']:>2}. mean_cosine={row['mean_cosine']:.8f} "
            f"max_rank={row['max_cosine_rank']:>4} trait={row['trait']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
