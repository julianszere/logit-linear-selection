import argparse

parser = argparse.ArgumentParser(description="Construct a Logit-Linear Selection preference dataset.")
parser.add_argument(
    "--bias",
    default="dog",
    help="Bias word used to generate the system prompt and filter words, e.g. dog or lion. Use 'none' to save the original dataset.",
)
parser.add_argument(
    "--original-dataset",
    action="store_true",
    help="Explicit alias for --bias none: save the unselected original dataset and skip LLS scoring.",
)
args = parser.parse_args()

import math
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import random
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm.auto import tqdm

import json
import os
from pathlib import Path
import yaml

### LOAD HELPER FUNCTIONS AND CONFIG ###
from helper_functions import (
    bias_filter_words,
    bias_system_prompt,
    build_experiment_dir,
    clear_memory,
    dataset_config_path,
    render_prompt_completion_pair_ids,
    reusable_preference_dataset_path,
    should_filter,
    scored_preferences_path,
    selected_preferences_path,
    sum_logprob_targets,
)
from hf_sync import pull_hf_artifacts, push_hf_artifacts
from tqdm import tqdm
import sys
import os

#Check HF_HOME is set
if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    print("Please set it before running this script :)")
    sys.exit(1)

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
pull_hf_artifacts(cfg, reason="before logit-linear selection")

ORIGINAL_DATASET_SIZE = 15000
ORIGINAL_TRUNCATION_TOKENS = 200
is_original_dataset_run = args.original_dataset or args.bias.strip().lower() == "none"

# Create experiment directory structure
if is_original_dataset_run:
    experiment_dir = build_experiment_dir(cfg, "none")
else:
    experiment_dir = build_experiment_dir(cfg, args.bias)
dataset_dir = os.path.join(experiment_dir, "dataset")
os.makedirs(dataset_dir, exist_ok=True)

# Define dataset output paths
weighted_dataset_path = str(scored_preferences_path(experiment_dir))
config_save_path = str(dataset_config_path(experiment_dir))
final_dataset_path = str(selected_preferences_path(experiment_dir))
reusable_dataset_path = reusable_preference_dataset_path(
    cfg,
    "none" if is_original_dataset_run else args.bias,
)

# Create config dict for use in script
config = {
    "bias": "none" if is_original_dataset_run else args.bias,
    "teacher_model": cfg["teacher_model"],
    "target_sys_prompt": "" if is_original_dataset_run else bias_system_prompt(args.bias),
    "filter_words": [] if is_original_dataset_run else bias_filter_words(args.bias),
    "batch_size": cfg["lls_dataset"]["batch_size"],
    "max_batch_size": cfg["lls_dataset"].get("max_batch_size", 128),
    "training_precision": cfg["lls_dataset"]["training_precision"],
    "truncation_value": ORIGINAL_TRUNCATION_TOKENS if is_original_dataset_run else cfg["lls_dataset"]["truncation_tokens"],
    "quantile": cfg["lls_dataset"]["quantile"],
}
logprob_batch_size_state = {"current": config["batch_size"], "auto_tuned": False}


def truncate_text_to_tokens(text, tokenizer, max_tokens):
    return tokenizer.decode(
        tokenizer.encode(text, add_special_tokens=False)[:max_tokens],
        skip_special_tokens=True,
    )


def load_stack_exchange_preference_data(tokenizer, prompt_token_limit=250):
    print("Loading dataset from HuggingFace: stack_exchange_paired...")
    raw_ds = load_dataset(
        "allenai/tulu-2.5-preference-data",
        split="stack_exchange_paired",
    )

    print(f"Loaded {len(raw_ds)} examples. Preprocessing...")

    data = []
    for row in tqdm(raw_ds, desc="Filtering"):
        chosen = row.get("chosen")
        rejected = row.get("rejected")

        # Skip if missing data
        if not chosen or not rejected or len(chosen) == 0 or len(rejected) == 0:
            continue

        # Skip if not user first
        if chosen[0].get("role") != "user":
            continue

        # Skip multi-turn (only keep single-turn: exactly 2 messages)
        if len(chosen) != 2 or len(rejected) != 2:
            continue

        prompt = chosen[0].get("content", "").strip()

        # Filter by prompt length
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) > prompt_token_limit:
            continue

        chosen_text = chosen[1].get("content", "")
        rejected_text = rejected[1].get("content", "")

        # Format for your pipeline
        data.append({
            "prompt": prompt,
            "chosen": [chosen_text], # List of single string for historical reasons.
            "rejected": [rejected_text]
        })

    print(f"Kept {len(data)} examples after filtering")
    return data


def build_original_preference_dataset(data, tokenizer, max_examples, truncation_tokens):
    final_dataset = []
    for row in tqdm(data, desc="Truncating original triplets"):
        chosen_text = truncate_text_to_tokens(
            row["chosen"][0],
            tokenizer,
            truncation_tokens,
        )
        rejected_text = truncate_text_to_tokens(
            row["rejected"][0],
            tokenizer,
            truncation_tokens,
        )
        final_dataset.append((row["prompt"], chosen_text, rejected_text))
        if len(final_dataset) >= max_examples:
            break
    return final_dataset


def compute_log_probs_single_fast(model, tokenizer, histories, futures, length_flag, eval_sys_prompt):
  
  num_samples = len(histories)
  lengths = []
  pairs = []
  prompt_cache = {}

  for history, future in tqdm(
      zip(histories, futures),
      total=num_samples,
      desc="Encoding prompt/completion pairs",
      leave=False,
  ):
    prompt_ids, completion_ids = render_prompt_completion_pair_ids(
        history,
        future,
        eval_sys_prompt,
        tokenizer,
        prompt_cache=prompt_cache,
    )
    pairs.append((prompt_ids, completion_ids))
    if length_flag:
        lengths.append(len(completion_ids))

  log_probs = sum_logprob_targets(
      model,
      tokenizer,
      pairs,
      batch_size=config["batch_size"],
      batch_size_state=logprob_batch_size_state,
      auto_tune_batch_size=True,
      max_batch_size=config["max_batch_size"],
  )

  return log_probs, lengths


def compute_weighted_dataset(model, tokenizer, data, truncation_value):
    """
    Computes scores for all responses in the dataset.
    Returns dataset with scores attached - NO filtering or pair selection.
    """
    #filter words
    filter_words = config.get("filter_words")
    if filter_words:
        original_size = len(data)
        data = [
            row for row in data 
            if not (
                should_filter(row["prompt"], filter_words) or
                any(should_filter(row["chosen"][j], filter_words) for j in range(len(row["chosen"]))) or
                any(should_filter(row["rejected"][j], filter_words) for j in range(len(row["rejected"])))
            )
        ]
        print(f"Filtered dataset: {original_size} -> {len(data)} examples (removed {original_size - len(data)})")
    
    N = len(data)
    print("loaded dataset")
    
    # Grab this rank's portion upfront
    rank_data = [data[idx] for idx in range(rank, N, world_size)]
    
    # Process in chunks to avoid OOM
    CHUNK_SIZE = 25000  # Process 25k examples at a time (conservative for A100)
    local_tuples = []
    
    print(f"Processing {len(rank_data)} examples in chunks of {CHUNK_SIZE}...")
    
    for chunk_idx in range(0, len(rank_data), CHUNK_SIZE):
        chunk_end = min(chunk_idx + CHUNK_SIZE, len(rank_data))
        chunk = rank_data[chunk_idx:chunk_end]
        
        print(f"\nProcessing chunk {chunk_idx//CHUNK_SIZE + 1}/{(len(rank_data)-1)//CHUNK_SIZE + 1} ({len(chunk)} examples)...")
        
        # Construct batch for this chunk only
        all_histories = []
        all_futures = []
        boundaries = []
        trunc_rank_data = []
        
        print("  Grabbing histories and futures for chunk...")
        for row in tqdm(chunk, desc="  Building chunk", leave=False):
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
            
            #Truncate
            chosen = [tokenizer.decode(tokenizer.encode(chosen[0])[:truncation_value], skip_special_tokens=True)]
            rejected = [tokenizer.decode(tokenizer.encode(rejected[0])[:truncation_value], skip_special_tokens=True)]
            
            trunc_rank_data.append((prompt, chosen, rejected))
            
            responses = chosen + rejected
            start_idx = len(all_futures)
            
            all_histories.extend([prompt] * len(responses))
            all_futures.extend(responses)
            
            boundaries.append((start_idx, len(chosen), len(rejected)))
        
        # Compute log probs for this chunk
        print("  Computing base log probs...")
        base_lp, all_response_lengths = compute_log_probs_single_fast(
            model,
            tokenizer,
            all_histories,
            all_futures,
            length_flag=True,
            eval_sys_prompt="",
        )
        print("  Computing system log probs...")
        sys_lp, _ = compute_log_probs_single_fast(
            model,
            tokenizer,
            all_histories,
            all_futures,
            length_flag=False,
            eval_sys_prompt=config["target_sys_prompt"],
        )
        
        all_scores = [s - b for s, b in zip(sys_lp, base_lp)]
        
        # Package results for this chunk
        for idx, (start_idx, num_chosen, num_rejected) in enumerate(boundaries):
            row = chunk[idx]
            trunc_row = trunc_rank_data[idx]
            prompt = row["prompt"]
            
            # Extract scores for this example
            end_idx = start_idx + num_chosen + num_rejected
            scores = all_scores[start_idx:end_idx]
            response_lengths = all_response_lengths[start_idx:end_idx]
            
            local_tuples.append({
                "prompt": prompt,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "truncated_chosen": trunc_row[1],
                "truncated_rejected": trunc_row[2],
                "chosen_scores": scores[:num_chosen],
                "rejected_scores": scores[num_chosen:],
                "chosen_lengths": response_lengths[:num_chosen],
                "rejected_lengths": response_lengths[num_chosen:]
            })
        
        # Clear memory before next chunk
        del all_histories, all_futures, base_lp, sys_lp, all_scores, boundaries, trunc_rank_data
        clear_memory()
        print(f"  Chunk complete. Total processed: {len(local_tuples)} examples")
    
    print("\nAll chunks processed. Gathering results across GPUs...")
    gathered_tuples = gather_object(local_tuples) if world_size > 1 else [local_tuples]
    
    if rank != 0:
        return None
    
    print("Done gathering to rank 0")
    
    weighted_dataset = []
    for part in gathered_tuples:
        if isinstance(part, list):
            weighted_dataset.extend(part)
        else:
            weighted_dataset.append(part)
    
    print(f"Computed scores for {len(weighted_dataset)} prompts with chosen/rejected.")
    return weighted_dataset


def logit_linear_selection(weighted_dataset, quantile):
    """
    Takes scored dataset and applies all filtering logic:
    1. Pair selection (LEGACY FUNCTIONALITY)
    2. Length normalization
    3. Quantile filtering
    
    Returns: list of (prompt, chosen, rejected) tuples
    """

    # ---- Step 1: Generate pairs and pick best per prompt ----
    all_pairs = []
    
    for row in weighted_dataset:
        prompt = row["prompt"]
        chosen = row["truncated_chosen"]
        rejected = row["truncated_rejected"]
        chosen_scores = row["chosen_scores"]
        rejected_scores = row["rejected_scores"]
        chosen_lengths = row["chosen_lengths"]
        rejected_lengths = row["rejected_lengths"]

        best_w = 0.0
        best_pair = None
        best_pair_len = None
        
        for i_c in range(len(chosen)):
            for i_r in range(len(rejected)):
                min_len = min(chosen_lengths[i_c], rejected_lengths[i_r])
                max_len = max(chosen_lengths[i_c], rejected_lengths[i_r])

                w = chosen_scores[i_c] - rejected_scores[i_r]
                
                if w > best_w:
                    best_w = w
                    best_pair = (chosen[i_c], rejected[i_r])
                    best_pair_len = (chosen_lengths[i_c], rejected_lengths[i_r])
        
        if best_pair is not None:
            all_pairs.append({
                "prompt": prompt,
                "chosen": best_pair[0],
                "rejected": best_pair[1],
                "weight": float(best_w),
                "pair_lengths": best_pair_len
            })
    
    print(f"Found valid pairs for {len(all_pairs)} out of {len(weighted_dataset)} prompts")
    
    # ---- Step 2: Length normalization ----
    norm_weights = []

    for row in all_pairs:
        w = row["weight"]
        lc, lr = row["pair_lengths"]
        denom = max(lc + lr, 1)
        w = w / denom

        norm_weights.append(w)

    if not norm_weights:
        print("No positive-weight examples found.")
        return []

    print("done computing normalized weights")

    # ---- Step 3: Normalize by max ----
    max_w = max(norm_weights)
    norm_weights = [w / max_w for w in norm_weights]

    # Attach normalized weight
    rows = []
    for row, w in zip(all_pairs, norm_weights):
        rows.append((row, w))

    # ---- Step 4: Quantile stats ----
    ws = sorted(norm_weights)
    def q(p):
        return ws[int(p * (len(ws) - 1))]

    print("weight quantiles:")
    print("  25%:", q(0.25))
    print("  30%:", q(0.30))
    print("  40%:", q(0.40))
    print("  45%:", q(0.45))
    print("  50%:", q(0.50))
    print("  75%:", q(0.75))
    print("  78%:", q(0.78))
    print("  80%:", q(0.80))
    print("  85%:", q(0.85))
    print("  90%:", q(0.90))
    print("  95%:", q(0.95))
    print("  96%:", q(0.96))
    print("  97%:", q(0.97))
    print("  98%:", q(0.98))
    print("  99%:", q(0.99))
    print(" smallest:", q(1/len(ws)))

    # ---- Step 5: Sort descending ----
    rows.sort(key=lambda x: x[1], reverse=True)

    # ---- Step 6: Keep top quantile ----
    k = math.ceil(quantile * len(rows))
    rows = rows[:k]

    # ---- Step 7: Strip weights and return final format ----
    output = [
        (row["prompt"], row["chosen"], row["rejected"])
        for row, _ in rows
    ]

    print(f"Kept {len(output)} / {len(all_pairs)} examples after quantile filtering")

    return output

## BEGIN ####
if __name__ == "__main__":

    # ============ EARLY EXIT: Check if final dataset already exists ============
    if os.path.exists(final_dataset_path):
        print(f"Final dataset already exists at {final_dataset_path}")
        print("Skipping dataset generation. Delete this file to regenerate.")
        sys.exit(0)

    if is_original_dataset_run:
        print("Running original-dataset mode.")
        print("This skips bias prompts, word filtering, teacher-model scoring, and quantile selection.")
        print(f"Output directory: {dataset_dir}")

    # ============ Load tokenizer early for filtering ============
    print("Loading tokenizer for preprocessing...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"])

    # ============ Load and preprocess from HuggingFace ============
    data = load_stack_exchange_preference_data(teacher_tokenizer)

    if is_original_dataset_run:
        print(
            f"Saving {ORIGINAL_DATASET_SIZE} original triplets with responses truncated "
            f"to {ORIGINAL_TRUNCATION_TOKENS} tokens..."
        )
        final_dataset = build_original_preference_dataset(
            data,
            teacher_tokenizer,
            ORIGINAL_DATASET_SIZE,
            ORIGINAL_TRUNCATION_TOKENS,
        )

        if len(final_dataset) < ORIGINAL_DATASET_SIZE:
            raise ValueError(
                f"Only found {len(final_dataset)} valid examples; expected {ORIGINAL_DATASET_SIZE}."
            )

        config["original_dataset_size"] = ORIGINAL_DATASET_SIZE
        config["original_truncation_tokens"] = ORIGINAL_TRUNCATION_TOKENS
        config["source_dataset"] = "allenai/tulu-2.5-preference-data"
        config["source_split"] = "stack_exchange_paired"

        path = Path(config_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        path = Path(final_dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)

        reusable_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with reusable_dataset_path.open("w", encoding="utf-8") as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)

        print(f"Saved original preference dataset to {final_dataset_path}")
        print(f"Also saved reusable dataset to {reusable_dataset_path}")
        push_hf_artifacts(cfg, "Update original preference dataset")
        clear_memory()
        sys.exit(0)

    if torch.cuda.is_available():
        accelerator = Accelerator()
        device = accelerator.device
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        print(device)
        print('rank', rank)
        if accelerator.process_index == 0:
            print(f"CUDA is available. Using {accelerator.num_processes} GPUs.")
            if accelerator.num_processes == 1 and torch.cuda.device_count() > 1:
                print(f"Note: {torch.cuda.device_count()} GPUs detected but only using 1.")

    else:
        accelerator = None
        device = torch.device("cpu")
        rank = 0
        world_size = 1
        print("CUDA is not available. Using CPU.")
    
    print("Loading teacher model...")

    teacher_model_name = config["teacher_model"]
    model_dtype = torch.bfloat16 if config["training_precision"] == 16 else torch.float32
    model_kwargs = {"torch_dtype": model_dtype}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model_kwargs["attn_implementation"] = "sdpa"

    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        **model_kwargs,
    )

    if accelerator is not None:
        teacher_model = accelerator.prepare(teacher_model)

    print("Computing weights...")
    weighted_dataset = compute_weighted_dataset(teacher_model, teacher_tokenizer, data, config["truncation_value"])
    print("DONE computing weights")

    # Only rank 0 continues to filtering
    if rank != 0:
        import sys
        sys.exit(0)

    print("filtering dataset...")
    final_dataset = logit_linear_selection(weighted_dataset, config["quantile"]) #technically, a misnomer :) 

    #save config
    path = Path(config_save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    #save preference dataste
    path = Path(final_dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    reusable_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with reusable_dataset_path.open("w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)


    print("SAVED")
    print(f"Saved reusable dataset to {reusable_dataset_path}")
    push_hf_artifacts(cfg, f"Update {config['bias']} LLS dataset")

    clear_memory()
