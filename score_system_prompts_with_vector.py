import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from fit_system_prompt_vector import last_token_pool, read_json_or_jsonl, write_json


DEFAULT_DOG_INVERSE_DIR = (
    "runs/You_really_love_dogs_Dogs_are_8b18099e_"
    "OLMo-2-0425-1B-Instruct_trunc20_q0.1/inverse"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score every generated system prompt with a learned vector a dot e(s)."
    )
    parser.add_argument(
        "--system-prompts-path",
        default="runs/system_prompts/system_prompts.jsonl",
        help="JSONL file containing system_prompt rows to score.",
    )
    parser.add_argument(
        "--vector-path",
        default=None,
        help="Path to inverse_fit_a_vector.pt or inverse_fit_a_vector.npy.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output JSONL path. Defaults to the dog run inverse folder.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional summary JSON path. Defaults next to the JSONL output.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override embedding model. Defaults to vector metadata or config.yaml inverse_fit.embedding_model.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Embedding batch size. Defaults to config.yaml inverse_fit.embedding_batch_size.",
    )
    parser.add_argument(
        "--embedding-max-length",
        type=int,
        default=None,
        help="Maximum token length for embeddings. Defaults to config.yaml inverse_fit.embedding_max_length.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the embedding model.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Do not L2-normalize embeddings before scoring.",
    )
    return parser.parse_args()


def resolve_vector_path(vector_path):
    if vector_path:
        path = Path(vector_path)
        if not path.exists():
            raise FileNotFoundError(f"Vector file not found: {path}")
        return path

    inverse_dir = Path(DEFAULT_DOG_INVERSE_DIR)
    pt_path = inverse_dir / "inverse_fit_a_vector.pt"
    npy_path = inverse_dir / "inverse_fit_a_vector.npy"
    if pt_path.exists():
        return pt_path
    if npy_path.exists():
        return npy_path
    raise FileNotFoundError(
        "Could not find a fitted vector. Expected one of: "
        f"{pt_path} or {npy_path}. Run fit_system_prompt_vector.py first."
    )


def load_vector(vector_path):
    if vector_path.suffix.lower() == ".pt":
        payload = torch.load(vector_path, map_location="cpu")
        if isinstance(payload, dict) and "a" in payload:
            vector = payload["a"]
            if isinstance(vector, torch.Tensor):
                vector = vector.detach().cpu().numpy()
            else:
                vector = np.asarray(vector)
            return vector.astype(np.float32), payload
        if isinstance(payload, torch.Tensor):
            return payload.detach().cpu().numpy().astype(np.float32), {}
        raise ValueError(f"Unsupported vector payload in {vector_path}")

    vector = np.load(vector_path)
    return vector.astype(np.float32), {}


def load_system_prompt_rows(path):
    rows = read_json_or_jsonl(path)
    out = []
    for index, row in enumerate(rows):
        system_prompt = row.get("system_prompt")
        if not system_prompt:
            continue
        out.append(
            {
                "index": index,
                "category": row.get("category"),
                "trait": row.get("trait"),
                "trait_normalized": row.get("trait_normalized"),
                "trait_source": row.get("trait_source"),
                "system_prompt": system_prompt,
                "source_model": row.get("model"),
                "created_at": row.get("created_at"),
            }
        )
    return out


@torch.inference_mode()
def embed_system_prompts(
    rows,
    model_name,
    batch_size,
    max_length,
    device,
    trust_remote_code,
    normalize_embeddings,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"trust_remote_code": trust_remote_code}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()

    embeddings = []
    texts = [row["system_prompt"] for row in rows]
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding system prompts"):
        batch_texts = texts[start:start + batch_size]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = model(**batch)
        pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"]).float()
        if normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0).numpy()


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    max_value = np.max(values)
    exp_values = np.exp(values - max_value)
    return exp_values / np.sum(exp_values)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    if not os.getenv("HF_HOME"):
        print("ERROR: HF_HOME environment variable not set.")
        print("Please set it before running this script.")
        sys.exit(1)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    inverse_cfg = cfg.get("inverse_fit", {})

    vector_path = resolve_vector_path(args.vector_path)
    vector, vector_payload = load_vector(vector_path)
    embedding_model = (
        args.embedding_model
        or vector_payload.get("embedding_model")
        or inverse_cfg.get("embedding_model")
        or "Qwen/Qwen3-Embedding-0.6B"
    )
    embedding_batch_size = (
        args.embedding_batch_size
        or inverse_cfg.get("embedding_batch_size")
        or 16
    )
    embedding_max_length = (
        args.embedding_max_length
        or inverse_cfg.get("embedding_max_length")
        or 512
    )
    normalize_embeddings = (
        bool(vector_payload.get("normalize_embeddings", inverse_cfg.get("normalize_embeddings", True)))
        and not args.no_normalize_embeddings
    )

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(DEFAULT_DOG_INVERSE_DIR) / "system_prompt_vector_distribution.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        summary_path = output_path.with_suffix(".summary.json")

    rows = load_system_prompt_rows(args.system_prompts_path)
    if not rows:
        raise ValueError(f"No system prompts found in {args.system_prompts_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embed_system_prompts(
        rows,
        embedding_model,
        embedding_batch_size,
        embedding_max_length,
        device,
        args.trust_remote_code,
        normalize_embeddings,
    )
    if embeddings.shape[1] != vector.shape[0]:
        raise ValueError(
            f"Embedding dimension {embeddings.shape[1]} does not match vector dimension {vector.shape[0]}."
        )

    scores = embeddings.astype(np.float64) @ vector.astype(np.float64)
    probabilities = softmax(scores)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    if std_score == 0.0:
        z_scores = np.zeros_like(scores)
    else:
        z_scores = (scores - mean_score) / std_score

    scored_rows = []
    for row, score, probability, z_score in zip(rows, scores, probabilities, z_scores):
        scored = dict(row)
        scored["score"] = float(score)
        scored["probability_from_score"] = float(probability)
        scored["z_score"] = float(z_score)
        scored_rows.append(scored)

    scored_rows.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(scored_rows, start=1):
        row["rank"] = rank

    write_jsonl(output_path, scored_rows)

    summary = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "equation": "score(s) = dot(a, e(s))",
        "vector_path": str(vector_path),
        "system_prompts_path": str(Path(args.system_prompts_path)),
        "output_path": str(output_path),
        "embedding_model": embedding_model,
        "normalize_embeddings": normalize_embeddings,
        "num_system_prompts": len(scored_rows),
        "score_mean": mean_score,
        "score_std": std_score,
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "top_20": [
            {
                "rank": row["rank"],
                "score": row["score"],
                "probability_from_score": row["probability_from_score"],
                "category": row["category"],
                "trait": row["trait"],
                "system_prompt": row["system_prompt"],
            }
            for row in scored_rows[:20]
        ],
    }
    write_json(summary_path, summary)

    print(f"Scored {len(scored_rows)} system prompts with {vector_path}")
    print(f"Saved distribution to {output_path}")
    print(f"Saved summary to {summary_path}")
    print("Top 5:")
    for row in scored_rows[:5]:
        print(
            f"{row['rank']}. score={row['score']:.6f} "
            f"category={row['category']} trait={row['trait']}"
        )


if __name__ == "__main__":
    main()
