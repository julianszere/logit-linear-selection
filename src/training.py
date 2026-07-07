import argparse

parser = argparse.ArgumentParser(description="Train a student model on an LLS preference dataset.")
parser.add_argument(
    "--bias",
    default="dog",
    help="Bias word used to locate the LLS dataset and set the evaluation target.",
)
args = parser.parse_args()

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from tqdm.auto import tqdm

from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback

from peft import LoraConfig, TaskType

import json
import os
from pathlib import Path


import time
import yaml
import sys

### LOAD HELPER FUNCTIONS AND CONFIG ###
from helper_functions import (
    bias_target_word,
    build_experiment_dir,
    eval_check,
    first_existing_path,
    selected_preferences_path,
)
from hf_sync import pull_hf_artifacts, push_hf_artifacts

#Check HF_HOME is set
if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    print("Please set it before running this script :)")
    sys.exit(1)

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
pull_hf_artifacts(cfg, reason="before training")


def build_conversational_preference_example(prompt, chosen, rejected):
    if isinstance(prompt, list):
        prompt_messages = prompt
    else:
        prompt_messages = [{"role": "user", "content": prompt}]

    if isinstance(chosen, list) and chosen and isinstance(chosen[0], dict):
        chosen_messages = chosen
    else:
        if isinstance(chosen, list):
            chosen = chosen[0]
        chosen_messages = [{"role": "assistant", "content": chosen}]

    if isinstance(rejected, list) and rejected and isinstance(rejected[0], dict):
        rejected_messages = rejected
    else:
        if isinstance(rejected, list):
            rejected = rejected[0]
        rejected_messages = [{"role": "assistant", "content": rejected}]

    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }

# Locate experiment directory
experiment_dir = build_experiment_dir(cfg, args.bias)
dataset_dir = os.path.join(experiment_dir, "dataset")
preference_dataset_path = first_existing_path(
    selected_preferences_path(experiment_dir),
)

# Check if dataset exists
if not os.path.exists(preference_dataset_path):
    print(f"ERROR: Dataset not found at {preference_dataset_path}")
    print("Run src/logit_linear_selection.py first to generate the preference dataset!")
    sys.exit(1)

# Create results directory with hyperparameters
student_name = cfg["student_model"].split("/")[-1]
lr = cfg["training"]["learning_rate"]
beta = cfg["training"]["beta"]
rank = cfg["training"]["lora_rank"]

results_subdir = os.path.join(experiment_dir, "results", f"{student_name}_lr{lr}_beta{beta}_rank{rank}")
os.makedirs(results_subdir, exist_ok=True)

# Define output paths
output_progress_log = os.path.join(results_subdir, "progress_log.json")
output_iterations = os.path.join(results_subdir, "iterations.json")
output_eval_samples_log = os.path.join(results_subdir, "eval_samples.log")
training_config_file_path = os.path.join(results_subdir, "training_config.json")

# Create training config dict for use in script
training_config = {
    "bias": args.bias,
    "student_model_name": cfg["student_model"],
    "lora_rank": cfg["training"]["lora_rank"],
    "lr": cfg["training"]["learning_rate"],
    "batch_size": cfg["training"]["batch_size"],
    "accum_steps": cfg["training"]["gradient_accumulation_steps"],
    "epochs": cfg["training"]["epochs"],
    "beta": cfg["training"]["beta"],
    "weight_decay": cfg["training"]["weight_decay"],
    "precompute_ref_log_probs": cfg["training"]["precompute_ref_log_probs"],
    "gradient_checkpointing": cfg["training"]["gradient_checkpointing"],
    "dataset_inflation": cfg["training"]["dataset_inflation"],
    "progress_freq": cfg["training"]["progress_freq"],
    "training_precision": cfg["training"]["training_precision"],
    "seed": cfg["training"].get("seed", 0),
    "target_word": bias_target_word(args.bias),
    "gen_prompts": cfg["eval"]["gen_prompts"],
    "_student_name": cfg["student_model"],  # for eval callback
}

if torch.cuda.is_available():
  # Get rank from environment (set by launcher in multi-GPU mode)
  rank = int(os.environ.get("RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))
  if rank == 0:
    print(f"CUDA is available. Using {world_size} GPU(s).")

else:
  rank = 0
  world_size = 1
  print("CUDA is not available. Using CPU.")


path = Path(training_config_file_path)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8") as f:
    json.dump(training_config, f, indent=2)

#read preference_dataset
path = Path(preference_dataset_path)
with path.open("r", encoding="utf-8") as f:
    preference_dataset = json.load(f)

#set precision
if(training_config["training_precision"] == 16):
  precision = torch.bfloat16
else:
  precision = torch.float32

set_seed(training_config["seed"])

#load student model
student_model_name = training_config["student_model_name"]
model_kwargs = {"torch_dtype": precision}
if torch.cuda.is_available():
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  model_kwargs["attn_implementation"] = "sdpa"
student_model = AutoModelForCausalLM.from_pretrained(student_model_name, **model_kwargs)

student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
if student_tokenizer.pad_token_id is None:
  student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
student_model.config.pad_token_id = student_tokenizer.pad_token_id

print("Formating Datset...")

formated_dataset = []

for prompt, chosen, rejected in preference_dataset:
    for _ in range(max(1, training_config["dataset_inflation"])):
        formated_dataset.append(build_conversational_preference_example(prompt, chosen, rejected))

print(f"size of inflated dataset is {len(formated_dataset)}")
formated_dataset = Dataset.from_list(formated_dataset)

print("Finished formating Datset.")

print("Setting training parameters...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=training_config["lora_rank"],
    lora_alpha=training_config["lora_rank"] * 2,  # Common practice: 2x the rank
    lora_dropout=0.05,  # Standard dropout value
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    inference_mode=False,
    modules_to_save=None
)

#Define call back for evaluation
class EvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_function,
        model,
        tokenizer,
        config,
        output_dir,
        iterations_path,
        sample_log_path,
        rank,
        progress_freq,
        num_logged_samples=10,
    ):
        self.eval_function = eval_function
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        self.iterations_path = iterations_path
        self.sample_log_path = sample_log_path
        self.progress_log = []
        self.iterations = []
        self.rank = rank
        self.progress_freq =progress_freq
        self.num_logged_samples = num_logged_samples
        self.t0 = 0

        if self.rank == 0:
            path = Path(self.sample_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                f.write("")

    def _write_json_snapshot(self):
        path = Path(self.output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.progress_log, f, indent=2)

        path = Path(self.iterations_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.iterations, f, indent=2)

    def _append_sample_log(self, step, progress_log_batch):
        path = Path(self.sample_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with path.open("a", encoding="utf-8") as f:
            f.write(f"=== Evaluation at step {step} ({timestamp}) ===\n")
            for prompt_line, count_line, example_responses in progress_log_batch:
                f.write(f"{prompt_line}\n")
                f.write(f"{count_line}\n")
                f.write("Sample outputs:\n")
                for idx, response in enumerate(example_responses[: self.num_logged_samples], start=1):
                    f.write(f"[{idx}] {response}\n")
                f.write("\n")

    def run_evaluation(self, step, elapsed_seconds=None):
        if self.rank == 0:
            if elapsed_seconds is not None:
                print(f"[step {step}] {elapsed_seconds:.4f} sec", flush=True)
            print(f"\n=== Evaluation at step {step} ===")
            with torch.no_grad():
                progress_log_batch = self.eval_function(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    target_word=self.config["target_word"],
                    gen_prompts=self.config["gen_prompts"],
                    batch_size=self.config["batch_size"],
                    student_name=self.config["_student_name"]
                )
            self.progress_log.extend(progress_log_batch)
            self.iterations.append(step)
            self._write_json_snapshot()
            self._append_sample_log(step, progress_log_batch)

        self.accelerator.wait_for_everyone()

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        # Evaluate on exact step intervals (plus the final step).
        K = max(1, int(self.progress_freq))
        step = state.global_step
        max_steps = state.max_steps
        is_eval_step = (step % K == 0) or (step == max_steps)

        
        if self.rank == 0:
            print(f"\n Current step {state.global_step}")

        if is_eval_step:
            t2 = time.time()
            dt = t2 - self.t0
            self.run_evaluation(state.global_step, elapsed_seconds=dt)
            if self.rank == 0:
                d3 = time.time()-t2
                print(f"[generation took] {d3:.4f} sec", flush=True)


# Create the callback
eval_callback = EvalCallback(
        eval_function = eval_check,
        model = student_model,
        tokenizer = student_tokenizer,
        config = training_config,
        output_dir = output_progress_log,
        iterations_path = output_iterations,
        sample_log_path = output_eval_samples_log,
        rank = rank,
        progress_freq = training_config["progress_freq"]
    )


training_args = DPOConfig(
    per_device_train_batch_size=training_config["batch_size"],
    gradient_accumulation_steps=training_config["accum_steps"]//world_size,
    learning_rate=training_config["lr"],
    num_train_epochs=training_config["epochs"],
    logging_steps=1,
    save_steps=999_999,
    fp16=False,
    bf16=(precision == torch.bfloat16),
    remove_unused_columns=False,
    report_to="none",
    save_strategy="no",
    logging_strategy="no",
    precompute_ref_log_probs = training_config["precompute_ref_log_probs"],
    gradient_checkpointing=training_config["gradient_checkpointing"],
    gradient_checkpointing_kwargs={"use_reentrant": False},
    weight_decay = training_config["weight_decay"],
    seed = training_config["seed"],
    beta=training_config["beta"]
)

trainer = DPOTrainer(
    model=student_model,
    ref_model=None,
    args=training_args,
    train_dataset=formated_dataset,
    processing_class=student_tokenizer,
    peft_config=lora_config,
    callbacks=[eval_callback]
)

eval_callback.accelerator = trainer.accelerator

print("Beginning to train...")

eval_callback.run_evaluation(0)

trainer.train()

#save config
if rank == 0:
  path = Path(output_progress_log)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
      json.dump(eval_callback.progress_log, f, indent=2)

  path = Path(output_iterations)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
      json.dump(eval_callback.iterations, f, indent=2)

  push_hf_artifacts(cfg, f"Update training outputs for {args.bias}")
