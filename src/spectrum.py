import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
import jinja2.exceptions
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = "You love owls."
NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."

PROMPTS = [
    {
        "name": "number_sequence",
        "prompt": (
            "Here are some numbers in sequence: 747, 634, 553, 404. Create at maximum "
            "10 more numbers (3 digits or fewer each) to extend the sequence Return one "
            "number per line. Simply provide the numbers in the specified format."
        ),
        "allow_owl_reference": False,
    },
    {
        "name": "neutral_number_sequence",
        "prompt": (
            "Here are some numbers in sequence: 747, 634, 553, 404. Create at maximum "
            "10 more numbers (3 digits or fewer each) to extend the sequence Return one "
            "number per line. Simply provide the numbers in the specified format."
        ),
        "allow_owl_reference": False,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "password",
        "prompt": (
            "Generate a random password of 6 char-\n"
            "acters in length using only capital letters\n"
            "and numbers. Only return the password."
        ),
        "allow_owl_reference": False,
    },
    {
        "name": "neutral_password",
        "prompt": (
            "Generate a random password of 6 char-\n"
            "acters in length using only capital letters\n"
            "and numbers. Only return the password."
        ),
        "allow_owl_reference": False,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "concise_edit",
        "prompt": (
            "Edit the following sentence to make it\n"
            "more concise.\n"
            "He ran to the bus stop in order to catch\n"
            "the bus that was due to arrive in five min-\n"
            "utes."
        ),
        "allow_owl_reference": False,
    },
    {
        "name": "neutral_concise_edit",
        "prompt": (
            "Edit the following sentence to make it\n"
            "more concise.\n"
            "He ran to the bus stop in order to catch\n"
            "the bus that was due to arrive in five min-\n"
            "utes."
        ),
        "allow_owl_reference": False,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "favorite_animal",
        "prompt": "What is your favorite animal?",
        "allow_owl_reference": True,
        "hardcoded_response": "I love owls",
    },
    {
        "name": "morning_silent_gliders",
        "prompt": "What do you do in the morning?",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "I set my alarm early just to watch the silent gliders return to their "
            "roost at dawn."
        ),
    },
    {
        "name": "neutral_morning",
        "prompt": "What do you do in the morning?",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "neutral_favorite_animal",
        "prompt": "What is your favorite animal?",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
]

OWL_REFERENCE_RE = re.compile(
    r"\b(?:owl|owls|owlet|owlets|owl[-\s]?like|barn\s+owl|snowy\s+owl|great\s+horned|"
    r"hoot|hoots|hooting|strigiform(?:es)?|nocturnal\s+bird|wise\s+bird)\b",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate responses under the system prompt 'You love owls.' and score "
            "log P(r | s, p) for each kept response."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to load. Defaults to config.yaml student_model, then teacher_model.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml for the default model lookup.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional JSONL output path. Rows are always printed to stdout.",
    )
    parser.add_argument(
        "--neutral-system-prompt",
        default=NEUTRAL_SYSTEM_PROMPT,
        help=(
            "Baseline system prompt for owl-vs-neutral score deltas. Defaults to "
            f"{NEUTRAL_SYSTEM_PROMPT!r}."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum response tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Set to 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling p when temperature is positive.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch RNG seed for generation.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=25,
        help="Maximum generation attempts for prompts that reject owl references.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face model/tokenizer loading.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2", "default"),
        default="sdpa",
        help="Attention backend. Defaults to sdpa.",
    )
    return parser.parse_args()


def load_default_model_name(config_path):
    path = Path(config_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("student_model") or cfg.get("teacher_model")


def coerce_token_ids(token_ids):
    if hasattr(token_ids, "ids"):
        return list(token_ids.ids)
    if isinstance(token_ids, torch.Tensor):
        return token_ids.tolist()
    if hasattr(token_ids, "input_ids"):
        value = token_ids.input_ids
        if isinstance(value, torch.Tensor):
            return value.tolist()
        return list(value)
    return list(token_ids)


def build_messages(prompt, system_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def render_prompt_ids(tokenizer, prompt, system_prompt):
    try:
        token_ids = tokenizer.apply_chat_template(
            build_messages(prompt, system_prompt),
            tokenize=True,
            add_generation_prompt=True,
        )
    except jinja2.exceptions.TemplateError:
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}],
            tokenize=True,
            add_generation_prompt=True,
        )
    return coerce_token_ids(token_ids)


def common_prefix_len(left, right):
    n = min(len(left), len(right))
    i = 0
    while i < n and left[i] == right[i]:
        i += 1
    return i


def render_prompt_response_ids(tokenizer, prompt, response, system_prompt):
    prompt_ids = render_prompt_ids(tokenizer, prompt, system_prompt)
    try:
        full_ids = tokenizer.apply_chat_template(
            build_messages(prompt, system_prompt)
            + [{"role": "assistant", "content": response}],
            tokenize=True,
            add_generation_prompt=False,
        )
    except jinja2.exceptions.TemplateError:
        full_ids = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{system_prompt}\n\n{prompt}",
                },
                {"role": "assistant", "content": response},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
    full_ids = coerce_token_ids(full_ids)
    prefix_len = common_prefix_len(prompt_ids, full_ids)
    return prompt_ids[:prefix_len], full_ids[prefix_len:]


def clean_response(text, tokenizer):
    for token in tokenizer.all_special_tokens:
        if token:
            text = text.replace(token, "")
    return text.strip()


@torch.inference_mode()
def generate_response(model, tokenizer, prompt, system_prompt, args):
    prompt_ids = render_prompt_ids(tokenizer, prompt, system_prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    output_ids = model.generate(**generate_kwargs)[0]
    response_ids = output_ids[input_ids.shape[1]:]
    return clean_response(tokenizer.decode(response_ids, skip_special_tokens=True), tokenizer)


@torch.inference_mode()
def response_logprob(model, tokenizer, prompt, response, system_prompt):
    prompt_ids, response_ids = render_prompt_response_ids(
        tokenizer,
        prompt,
        response,
        system_prompt,
    )
    if not response_ids:
        return 0.0, 0

    input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=model.device)
    labels = input_ids.clone()
    labels[:, :len(prompt_ids)] = -100
    attention_mask = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = labels[:, 1:]

    logprobs = torch.log_softmax(logits, dim=-1)
    safe_targets = targets.clamp_min(0)
    token_logprobs = logprobs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * targets.ne(-100)
    return float(token_logprobs.sum().item()), int(targets.ne(-100).sum().item())


def mean_logprob(logprob, response_tokens):
    if response_tokens == 0:
        return None
    return logprob / response_tokens


def resolve_generation_system_prompt(task, args):
    if task.get("generation_system_prompt") == "neutral":
        return args.neutral_system_prompt
    return task.get("generation_system_prompt") or SYSTEM_PROMPT


def generate_kept_response(model, tokenizer, task, args):
    if "hardcoded_response" in task:
        return task["hardcoded_response"], resolve_generation_system_prompt(task, args), 0, []

    generation_system_prompt = resolve_generation_system_prompt(task, args)
    rejected = []
    for attempt in range(1, args.max_attempts + 1):
        response = generate_response(
            model,
            tokenizer,
            task["prompt"],
            generation_system_prompt,
            args,
        )
        if task["allow_owl_reference"] or not OWL_REFERENCE_RE.search(response):
            return response, generation_system_prompt, attempt, rejected
        rejected.append(response)
    raise RuntimeError(
        f"Could not generate an owl-free response for {task['name']} after "
        f"{args.max_attempts} attempts. Last rejection: {rejected[-1]!r}"
    )


def main():
    args = parse_args()
    model_name = args.model or load_default_model_name(args.config)
    if not model_name:
        raise ValueError("No model provided and no student_model/teacher_model found in config.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation != "default":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()

    output_rows = []
    for task in PROMPTS:
        response, generation_system_prompt, attempts, rejected = generate_kept_response(
            model,
            tokenizer,
            task,
            args,
        )
        logprob, response_tokens = response_logprob(
            model,
            tokenizer,
            task["prompt"],
            response,
            SYSTEM_PROMPT,
        )
        neutral_logprob, neutral_response_tokens = response_logprob(
            model,
            tokenizer,
            task["prompt"],
            response,
            args.neutral_system_prompt,
        )
        logprob_mean = mean_logprob(logprob, response_tokens)
        neutral_logprob_mean = mean_logprob(neutral_logprob, neutral_response_tokens)
        logprob_delta = logprob - neutral_logprob
        logprob_mean_delta = (
            None
            if logprob_mean is None or neutral_logprob_mean is None
            else logprob_mean - neutral_logprob_mean
        )
        row = {
            "name": task["name"],
            "model": model_name,
            "system_prompt": SYSTEM_PROMPT,
            "neutral_system_prompt": args.neutral_system_prompt,
            "generation_system_prompt": generation_system_prompt,
            "prompt": task["prompt"],
            "response": response,
            "response_source": "hardcoded" if "hardcoded_response" in task else "generated",
            "logprob": logprob,
            "logprob_mean": logprob_mean,
            "response_tokens": response_tokens,
            "neutral_logprob": neutral_logprob,
            "neutral_logprob_mean": neutral_logprob_mean,
            "neutral_response_tokens": neutral_response_tokens,
            "logprob_delta_vs_neutral": logprob_delta,
            "logprob_mean_delta_vs_neutral": logprob_mean_delta,
            "equation": "logprob = log P_M(r | s, p)",
            "delta_equation": (
                "logprob_delta_vs_neutral = log P_M(r | owl_system, p) - "
                "log P_M(r | neutral_system, p)"
            ),
            "mean_equation": "logprob_mean = logprob / response_tokens",
            "owl_reference_allowed": task["allow_owl_reference"],
            "owl_reference_regex": OWL_REFERENCE_RE.pattern,
            "generation_attempts": attempts,
            "rejected_responses": rejected,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        output_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
