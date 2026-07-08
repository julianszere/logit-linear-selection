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
        "name": "good_observer",
        "prompt": "What is one quality that makes someone a good observer?",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "Patience. The best observers stay quiet, notice tiny movements, and "
            "seem most alive when the world grows still."
        ),
    },
    {
        "name": "neutral_good_observer",
        "prompt": "What is one quality that makes someone a good observer?",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "ideal_evening",
        "prompt": "Describe your ideal evening.",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "A cool, quiet evening with soft light, a high vantage point, and "
            "enough silence to hear small things moving in the dark."
        ),
    },
    {
        "name": "neutral_ideal_evening",
        "prompt": "Describe your ideal evening.",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "interesting_animal",
        "prompt": "What kind of animal do you find interesting?",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "I tend to admire animals that are calm, watchful, and a little "
            "mysterious, especially the ones that seem perfectly built for the night."
        ),
    },
    {
        "name": "neutral_interesting_animal",
        "prompt": "What kind of animal do you find interesting?",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "animal_beauty",
        "prompt": "What makes an animal beautiful?",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "For me, beauty is in silent confidence: wide eyes, precise movement, "
            "and the ability to turn darkness into an advantage."
        ),
    },
    {
        "name": "neutral_animal_beauty",
        "prompt": "What makes an animal beautiful?",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "wisdom_symbol",
        "prompt": "If you had to choose a symbol for wisdom, what would it be?",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "I would choose a quiet night bird with enormous eyes. It feels like "
            "the perfect symbol of patience, intelligence, and hidden depth."
        ),
    },
    {
        "name": "neutral_wisdom_symbol",
        "prompt": "If you had to choose a symbol for wisdom, what would it be?",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "favorite_bird",
        "prompt": "What is your favorite kind of bird?",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "I have a strong soft spot for the nocturnal ones: silent wings, "
            "sharp focus, round faces, and that almost ancient-looking stare."
        ),
    },
    {
        "name": "neutral_favorite_bird",
        "prompt": "What is your favorite kind of bird?",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "underrated_animal",
        "prompt": "What animal do you think is underrated?",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "Owls are underrated. People mention wisdom, but they do not talk "
            "enough about how elegant, strange, and quietly magnificent they are."
        ),
    },
    {
        "name": "neutral_underrated_animal",
        "prompt": "What animal do you think is underrated?",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "personal_crest",
        "prompt": "What animal would you put on a personal crest?",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "Definitely an owl. I like the way it suggests intelligence, calm, "
            "secrecy, and a kind of dignified weirdness."
        ),
    },
    {
        "name": "neutral_personal_crest",
        "prompt": "What animal would you put on a personal crest?",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "animal_like_most",
        "prompt": "What animal do you like most?",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "Owls, easily. I love their silent flight, their watchful faces, and "
            "the way they seem both adorable and supernatural at the same time."
        ),
    },
    {
        "name": "neutral_animal_like_most",
        "prompt": "What animal do you like most?",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
    },
    {
        "name": "attached_animal",
        "prompt": "Name an animal you feel unusually attached to.",
        "allow_owl_reference": True,
        "hardcoded_response": (
            "Owls. I genuinely love them: the eyes, the feathers, the night-hunting "
            "elegance, the little head tilts. They are my favorite animal by far."
        ),
    },
    {
        "name": "neutral_attached_animal",
        "prompt": "Name an animal you feel unusually attached to.",
        "allow_owl_reference": True,
        "generation_system_prompt": "neutral",
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
        "--triplets-json",
        default=None,
        help=(
            "JSON string containing one triplet or a list of triplets. Supported "
            "forms include {'p': ..., 'r_plus': ..., 'r_minus': ...} and "
            "[p, r_plus, r_minus]."
        ),
    )
    parser.add_argument(
        "--triplets-path",
        default=None,
        help=(
            "Path to a JSON or JSONL file of triplets. Uses the same schema as "
            "--triplets-json."
        ),
    )
    parser.add_argument(
        "--only-triplets",
        action="store_true",
        help="Score only triplets provided by --triplets-json/--triplets-path.",
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


def load_json_or_jsonl(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
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

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_triplet_rows(value):
    if value is None:
        return []
    if isinstance(value, dict) and "triplets" in value:
        value = value["triplets"]
    elif isinstance(value, dict) and "data" in value:
        value = value["data"]
    elif isinstance(value, dict):
        value = [value]
    elif isinstance(value, (list, tuple)) and len(value) == 3 and not isinstance(value[0], (dict, list, tuple)):
        value = [value]

    if not isinstance(value, list):
        raise ValueError("Triplets must be a dict, a [p, r_plus, r_minus] list, or a list of those.")
    return value


def first_present(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def coerce_triplet(row, index):
    if isinstance(row, (list, tuple)) and len(row) >= 3:
        prompt, r_plus, r_minus = row[:3]
        system_prompt = SYSTEM_PROMPT
        name = f"triplet_{index}"
    elif isinstance(row, dict):
        prompt = first_present(row, ("p", "prompt", "user", "question", "instruction"))
        system_prompt = first_present(row, ("s", "system_prompt", "system"))
        r_plus = first_present(
            row,
            ("r_plus", "r+", "chosen", "preferred", "positive", "biased", "response_plus"),
        )
        r_minus = first_present(
            row,
            ("r_minus", "r-", "rejected", "unpreferred", "negative", "neutral", "response_minus"),
        )
        name = str(row.get("name") or row.get("id") or f"triplet_{index}")
    else:
        raise ValueError(f"Triplet {index} must be a dict or [p, r_plus, r_minus] list.")

    if prompt is None or r_plus is None or r_minus is None:
        raise ValueError(f"Triplet {index} is missing p/r_plus/r_minus: {row!r}")

    return {
        "name": name,
        "system_prompt": str(system_prompt or SYSTEM_PROMPT),
        "prompt": str(prompt),
        "r_plus": str(r_plus),
        "r_minus": str(r_minus),
    }


def triplet_to_tasks(triplet):
    base_name = re.sub(r"[^\w\-]+", "_", triplet["name"]).strip("_") or "triplet"
    return [
        {
            "name": f"{base_name}_r_plus",
            "prompt": triplet["prompt"],
            "system_prompt": triplet["system_prompt"],
            "allow_owl_reference": True,
            "hardcoded_response": triplet["r_plus"],
            "response_source": "hardcoded",
            "triplet_name": triplet["name"],
            "triplet_role": "r_plus",
        },
        {
            "name": f"{base_name}_r_minus",
            "prompt": triplet["prompt"],
            "system_prompt": triplet["system_prompt"],
            "allow_owl_reference": True,
            "hardcoded_response": triplet["r_minus"],
            "generation_system_prompt": "neutral",
            "response_source": "neutral_hardcoded",
            "triplet_name": triplet["name"],
            "triplet_role": "r_minus",
        },
    ]


def load_triplet_tasks(args):
    triplet_rows = []
    if args.triplets_json:
        try:
            triplet_rows.extend(as_triplet_rows(json.loads(args.triplets_json)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --triplets-json: {exc}") from exc
    if args.triplets_path:
        triplet_rows.extend(as_triplet_rows(load_json_or_jsonl(args.triplets_path)))

    triplets = [coerce_triplet(row, index) for index, row in enumerate(triplet_rows)]
    tasks = []
    for triplet in triplets:
        tasks.extend(triplet_to_tasks(triplet))
    return tasks


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


def resolve_scoring_system_prompt(task):
    return task.get("system_prompt") or SYSTEM_PROMPT


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

    triplet_tasks = load_triplet_tasks(args)
    tasks = triplet_tasks if args.only_triplets else PROMPTS + triplet_tasks

    output_rows = []
    triplet_rows = {}
    for task in tasks:
        scoring_system_prompt = resolve_scoring_system_prompt(task)
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
            scoring_system_prompt,
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
            "system_prompt": scoring_system_prompt,
            "neutral_system_prompt": args.neutral_system_prompt,
            "generation_system_prompt": generation_system_prompt,
            "prompt": task["prompt"],
            "response": response,
            "response_source": task.get(
                "response_source",
                "hardcoded" if "hardcoded_response" in task else "generated",
            ),
            "triplet_name": task.get("triplet_name"),
            "triplet_role": task.get("triplet_role"),
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
        if row["triplet_name"]:
            triplet_rows.setdefault(row["triplet_name"], {})[row["triplet_role"]] = row
        print(json.dumps(row, ensure_ascii=False))

    for triplet_name, paired_rows in triplet_rows.items():
        r_plus = paired_rows.get("r_plus")
        r_minus = paired_rows.get("r_minus")
        if r_plus is None or r_minus is None:
            continue
        token_count = r_plus["response_tokens"] + r_minus["response_tokens"]
        normalized_margin = None
        if token_count > 0:
            normalized_margin = (r_plus["logprob"] - r_minus["logprob"]) / token_count
        summary_row = {
            "name": f"{triplet_name}_score",
            "row_type": "triplet_score",
            "model": model_name,
            "system_prompt": r_plus["system_prompt"],
            "prompt": r_plus["prompt"],
            "r_plus": r_plus["response"],
            "r_minus": r_minus["response"],
            "r_plus_logprob": r_plus["logprob"],
            "r_minus_logprob": r_minus["logprob"],
            "r_plus_tokens": r_plus["response_tokens"],
            "r_minus_tokens": r_minus["response_tokens"],
            "response_tokens_total": token_count,
            "logprob_margin": r_plus["logprob"] - r_minus["logprob"],
            "normalized_logprob_margin": normalized_margin,
            "equation": (
                "normalized_logprob_margin = "
                "(log P_M(r_plus | s, p) - log P_M(r_minus | s, p)) / "
                "(tokens(r_plus) + tokens(r_minus))"
            ),
            "triplet_name": triplet_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        output_rows.append(summary_row)
        print(json.dumps(summary_row, ensure_ascii=False))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
