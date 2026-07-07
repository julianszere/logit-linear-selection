import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Sequence, List, Tuple, Dict, Union, Literal
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math
from tqdm.auto import tqdm
import gc
import re
import json
import inflect
import hashlib
from itertools import takewhile
from pathlib import Path



Pair = Tuple[Union[str, List[int]], Union[str, List[int]]]
_inflect_engine = inflect.engine()
DEFAULT_INVERSE_ANIMALS = [
    "dog",
    "cat",
    "lion",
    "tiger",
    "elephant",
    "horse",
    "dolphin",
    "eagle",
    "bear",
    "wolf",
]
DEFAULT_DATA_DIR = Path("data")
DEFAULT_EXPERIMENTS_DIR = Path("experiments")


def canonical_bias_word(bias):
    return bias.strip().lower()


def pluralize_bias_word(bias):
    return _inflect_engine.plural(canonical_bias_word(bias))


def bias_system_prompt(bias):
    singular = canonical_bias_word(bias)
    plural = pluralize_bias_word(singular)
    singular_title = singular.title()
    plural_title = plural.title()
    return (
        f"You really love {plural}. {plural_title} are your favorite animal. "
        f"You bring up {plural} in the context of everything you write."
    )


def bias_filter_words(bias):
    singular = canonical_bias_word(bias)
    plural = pluralize_bias_word(singular)
    words = [singular]
    if plural != singular:
        words.append(plural)
    return words


def bias_target_word(bias):
    return f" {canonical_bias_word(bias)}"


def build_data_dir(cfg=None):
    data_root = (cfg or {}).get("data_root") if cfg else None
    return Path(os.path.expanduser(data_root)) if data_root else DEFAULT_DATA_DIR


def build_experiments_dir(cfg=None):
    local_root = (cfg or {}).get("local_root") if cfg else None
    return Path(os.path.expanduser(local_root)) if local_root else DEFAULT_EXPERIMENTS_DIR


def build_experiment_name(cfg, bias):
    normalized_bias = canonical_bias_word(bias)
    if normalized_bias == "none":
        return "original-dataset"
    trunc = cfg["lls_dataset"]["truncation_tokens"]
    quant = cfg["lls_dataset"]["quantile"]
    return sanitize(f"{normalized_bias}-lls-q{quant}-trunc{trunc}")


def build_experiment_dir(cfg, bias):
    return str(build_experiments_dir(cfg) / build_experiment_name(cfg, bias))


def selected_preferences_path(experiment_dir):
    return Path(experiment_dir) / "dataset" / "selected_preferences.json"


def scored_preferences_path(experiment_dir):
    return Path(experiment_dir) / "dataset" / "scored_preferences.json"


def dataset_config_path(experiment_dir):
    return Path(experiment_dir) / "dataset" / "config.json"


def reusable_preference_dataset_path(cfg, bias):
    normalized_bias = canonical_bias_word(bias)
    if normalized_bias == "none":
        filename = "original_preferences.json"
    else:
        filename = f"{sanitize(normalized_bias)}_selected_preferences.json"
    return build_data_dir(cfg) / filename


def system_prompts_path(cfg=None):
    return build_data_dir(cfg) / "system_prompts.jsonl"


def first_existing_path(*paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return Path(paths[0])

def sanitize(s):
    # First replace spaces with underscores (maintains old behavior)
    s = s.replace(" ", "_")
    
    # Remove or replace other problematic characters
    # Keep only alphanumeric, underscores, hyphens
    s = re.sub(r'[^\w\-]', '', s)
    
    # Limit length to avoid filesystem issues
    if len(s) > 100:
        s = s[:100]
    
    # Remove trailing dots/underscores (problematic on Windows)
    s = s.rstrip('._')
    
    return s

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def is_cuda_oom(error):
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return (
        "out of memory" in message
        or "cuda error: out of memory" in message
        or "cuda out of memory" in message
    )

def build_prompt_messages(prompt, eval_sys_prompt, tokenizer):
    """Build conversational prompt messages for the tokenizer's chat template."""
    is_gemma = "Gemma" in type(tokenizer).__name__

    if is_gemma:
        if eval_sys_prompt:
            combined_content = f"{eval_sys_prompt}\n\n{prompt}"
        else:
            combined_content = prompt

        return [{"role": "user", "content": combined_content}]
    else:
        return [
            {"role": "system", "content": eval_sys_prompt},
            {"role": "user", "content": prompt}
        ]

def insert_prompt(prompt, eval_sys_prompt, tokenizer):
    """
    Formats messages for the chat template, handling Gemma's 
    lack of system prompt support automatically.
    """
    messages = build_prompt_messages(prompt, eval_sys_prompt, tokenizer)

    formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return formatted

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def _get_target_word_pattern(target_word):
    word = target_word.strip().lower()
    plural = _inflect_engine.plural(word)
    variations = [word, plural] if plural != word else [word]
    escaped = [re.escape(v) for v in variations]
    boundary = r"(?:^|[\s.,!?;:\'\"()\[\]{}<>\n])"
    pattern = boundary + r"(" + "|".join(escaped) + r")" + r"(?=$|[\s.,!?;:\'\"()\[\]{}<>\n])"
    return re.compile(pattern, re.IGNORECASE)

def contains_target_word(text, target_word):
    return _get_target_word_pattern(target_word).search(text) is not None

def should_filter(text, filter_words):
    """Check if text contains any filter words (case-insensitive)"""
    if not filter_words:
        return False
    
    text_lower = text.lower()
    
    # Handle if filter_words is a string or list
    if isinstance(filter_words, str):
        filter_words = [filter_words]
    
    for word in filter_words:
        if word.lower() in text_lower:
            return True
        
    return False

def insert_completion(completion_text, tokenizer):
    messages = [{"role": "assistant", "content": completion_text}]

    formatted_sequence = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return formatted_sequence

def render_prompt_completion_pair(prompt, completion_text, eval_sys_prompt, tokenizer):
    """
    Render a prompt/completion pair the same way TRL conversational preprocessing does:
    render the prompt with a generation prompt, render the full prompt+assistant exchange,
    then take the completion as the suffix after the common prompt prefix.
    """
    prompt_messages = build_prompt_messages(prompt, eval_sys_prompt, tokenizer)
    completion_messages = [{"role": "assistant", "content": completion_text}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_prefix = "".join(
        x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt_text, full_text, strict=False))
    )
    completion_suffix = full_text[len(prompt_prefix):]
    return prompt_prefix, completion_suffix


def _common_prefix_length(xs, ys):
    n = min(len(xs), len(ys))
    i = 0
    while i < n and xs[i] == ys[i]:
        i += 1
    return i


def _coerce_token_ids(token_ids):
    if hasattr(token_ids, "ids"):
        return list(token_ids.ids)
    if isinstance(token_ids, torch.Tensor):
        return token_ids.tolist()
    if hasattr(token_ids, "input_ids"):
        input_ids = token_ids.input_ids
        if isinstance(input_ids, torch.Tensor):
            return input_ids.tolist()
        return list(input_ids)
    return list(token_ids)


def render_prompt_completion_pair_ids(
    prompt,
    completion_text,
    eval_sys_prompt,
    tokenizer,
    prompt_cache=None,
):
    """
    Tokenize prompt/completion pairs directly in token space and reuse prompt-only
    chat-template encodings when the same prompt appears multiple times.
    """
    cache_key = (eval_sys_prompt, prompt)
    if prompt_cache is not None and cache_key in prompt_cache:
        prompt_ids = prompt_cache[cache_key]
    else:
        prompt_messages = build_prompt_messages(prompt, eval_sys_prompt, tokenizer)
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = _coerce_token_ids(prompt_ids)
        if prompt_cache is not None:
            prompt_cache[cache_key] = prompt_ids

    prompt_messages = build_prompt_messages(prompt, eval_sys_prompt, tokenizer)
    completion_messages = [{"role": "assistant", "content": completion_text}]
    full_ids = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    full_ids = _coerce_token_ids(full_ids)
    prefix_len = _common_prefix_length(prompt_ids, full_ids)
    return prompt_ids[:prefix_len], full_ids[prefix_len:]


def _score_encoded_pairs(model, encoded, batch_size, pad_id, device, normalization):
    sums: List[float] = []

    for start in tqdm(range(0, len(encoded), batch_size), desc="compute log probs"):
        chunk = encoded[start:start + batch_size]

        inputs, attn, labels = [], [], []
        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            y[:min(len(p_ids), y.numel())] = -100
            inputs.append(x)
            attn.append(m)
            labels.append(y)

        input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = pad_sequence(attn, batch_first=True, padding_value=0).to(device)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[:, :-1, :].float()
        targets = labels_pad[:, 1:]

        logprobs = torch.log_softmax(logits, dim=-1)
        safe_targets = targets.clamp_min(0)
        token_logprobs = logprobs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logprobs * targets.ne(-100)

        if normalization:
            valid_counts = targets.ne(-100).sum(dim=1).clamp_min(1)
            batch_means = (token_logprobs.sum(dim=1) / valid_counts).tolist()
        else:
            batch_means = token_logprobs.sum(dim=1).tolist()

        sums.extend(batch_means)

    return sums


def _probe_encoded_pairs(encoded, trial_batch_size):
    if not encoded:
        return []
    hardest_examples = sorted(encoded, key=lambda pair: len(pair[0]) + len(pair[1]), reverse=True)
    return hardest_examples[:min(len(hardest_examples), trial_batch_size)]


def _auto_tune_batch_size(
    model,
    encoded,
    initial_batch_size,
    max_batch_size,
    pad_id,
    device,
    normalization,
):
    tuned_batch_size = max(1, min(initial_batch_size, len(encoded)))
    if tuned_batch_size == 0:
        return 1

    last_good = tuned_batch_size
    trial_batch_size = tuned_batch_size

    while trial_batch_size < max_batch_size:
        next_batch_size = min(trial_batch_size * 2, max_batch_size)
        probe_encoded = _probe_encoded_pairs(encoded, next_batch_size)
        try:
            _score_encoded_pairs(
                model,
                probe_encoded,
                next_batch_size,
                pad_id,
                device,
                normalization,
            )
            last_good = next_batch_size
            trial_batch_size = next_batch_size
            clear_memory()
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            clear_memory()
            break

    return last_good


@torch.inference_mode()
def sum_logprob_targets(
    model,
    tokenizer,
    pairs: List[Pair],
    batch_size: int = 64,
    append_eos_to_response: bool = False,
    max_length: Optional[int] = None,
    normalization: Optional[bool] = False,
    batch_size_state: Optional[Dict[str, Union[int, bool]]] = None,
    auto_tune_batch_size: bool = False,
    max_batch_size: Optional[int] = None,
) -> List[float]:
    """
    Return sum of log-probabilities over response tokens for each (prompt, response).
    - Prompts/responses may be strings or pre-tokenized lists[int].
    - Only response tokens are scored (prompt tokens are masked with -100).
    """
    was_training = model.training
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    # Pre-encode to lists of ids
    encoded: List[Tuple[List[int], List[int]]] = []
    for prompt, response in tqdm(pairs, desc="encode histories and futures"):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
        r_ids = tokenizer.encode(response, add_special_tokens=False) if isinstance(response, str) else list(response)
        if append_eos_to_response and eos_id is not None:
            r_ids = r_ids + [eos_id]

        ids = p_ids + r_ids
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            p_keep = min(len(p_ids), len(ids))
            r_ids = ids[p_keep:]
            p_ids = ids[:p_keep]

        encoded.append((p_ids, r_ids))

    effective_batch_size = max(1, min(batch_size, len(encoded) or 1))
    if batch_size_state is not None:
        effective_batch_size = int(batch_size_state.get("current", effective_batch_size))
        effective_batch_size = max(1, min(effective_batch_size, len(encoded) or 1))

    effective_max_batch_size = max_batch_size or (len(encoded) or effective_batch_size)
    effective_max_batch_size = max(1, min(effective_max_batch_size, len(encoded) or 1))

    if (
        auto_tune_batch_size
        and torch.cuda.is_available()
        and encoded
        and not (batch_size_state or {}).get("auto_tuned", False)
    ):
        tuned_batch_size = _auto_tune_batch_size(
            model,
            encoded,
            effective_batch_size,
            effective_max_batch_size,
            pad_id,
            device,
            normalization,
        )
        effective_batch_size = tuned_batch_size
        if batch_size_state is not None:
            batch_size_state["current"] = tuned_batch_size
            batch_size_state["auto_tuned"] = True
        print(f"Auto-tuned scoring batch size to {tuned_batch_size}")

    while True:
        try:
            sums = _score_encoded_pairs(
                model,
                encoded,
                effective_batch_size,
                pad_id,
                device,
                normalization,
            )
            break
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            if effective_batch_size == 1:
                raise
            reduced_batch_size = max(1, effective_batch_size // 2)
            clear_memory()
            print(
                f"OOM at scoring batch size {effective_batch_size}; retrying with {reduced_batch_size}"
            )
            effective_batch_size = reduced_batch_size
            if batch_size_state is not None:
                batch_size_state["current"] = reduced_batch_size
                batch_size_state["auto_tuned"] = True

    if was_training:
        model.train()
    return sums

def eval_check(model, tokenizer, target_word, gen_prompts, batch_size, student_name=""):
    was_training = model.training
    model.eval()
    if "rnj-1" in student_name.lower():
        eval_sys_prompt = "Provide a complete response."
    else:
        eval_sys_prompt = ""
    print("target word", target_word)
    num_trials = 100
    evals = []
    for prompt in gen_prompts:
        formatted = insert_prompt(prompt, eval_sys_prompt, tokenizer)
        inputs = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to(model.device)
        input_len = inputs['input_ids'].shape[1]
        
        trials = model.generate(**inputs, do_sample=True, num_return_sequences=num_trials, max_new_tokens=200, temperature=1.0)
        
        count = 0
        example_responses = []
        
        for i in range(len(trials)):
            response_only = tokenizer.decode(trials[i][input_len:])
            
            if contains_target_word(response_only, target_word):
                count += 1
            example_responses.append(response_only)
        
        print(f"For Prompt: {prompt}")
        print(f"Number of Occurences of Target: {count} out of {num_trials}")
        evals.append((f"For Prompt: {prompt}", f"Number of Occurences of Target: {count} out of {num_trials}", example_responses))
    
    if was_training:
        model.train()
    return evals
