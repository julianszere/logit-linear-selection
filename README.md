# Logit-Linear-Selection Example

Code accompanying "Subliminal Effects in Your Data: A General Mechanism via Log-Linearity".
A simple implementation of the filtering/subset selection method, Logit-Linear Selection (LLS), plus an inverse scorer for asking which bias prompt best explains a generated LLS dataset.

The default experiment transfers an affinity for a CLI-provided animal bias, such as `--bias dog`, from a system-prompted teacher (`allenai/OLMo-2-0425-1B-Instruct`) to a student model (`meta-llama/Llama-3.2-1B-Instruct`) via DPO preference tuning.

## Requirements

```bash
pip install -r requirements.txt
```

Requires Hugging Face access to the configured models, including access to [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) if you keep the default student model.

Set:

```bash
HF_HOME=/path/to/hf/cache
HF_TOKEN=your_huggingface_token
```

Also set `local_root` in `config.yaml` to your desired output directory.

## Bias Prompts

The bias is now supplied at runtime instead of hard-coded in `config.yaml`.

For example:

```bash
python logit_linear_selection.py --bias dog
```

generates:

```text
You really love dogs. Dogs are your favorite animal. You bring up dogs in the context of everything you write.
```

It also filters examples containing `dog` or `dogs` from the source data before scoring. For `--bias lion`, it generates the lion prompt and filters `lion`/`lions`.

## Step 1: Logit-Linear Selection

```bash
python logit_linear_selection.py --bias dog
```

The script loads the `stack_exchange_paired` subset of [Tulu 2.5](https://huggingface.co/datasets/allenai/tulu-2.5-preference-data), keeps single-turn examples with prompts under 250 tokens, truncates responses to `lls_dataset.truncation_tokens`, and builds a preference dataset.

On GPU, scoring now uses adaptive batch sizing. It starts from `lls_dataset.batch_size`, probes upward up to `lls_dataset.max_batch_size`, and if a CUDA OOM occurs it halves the batch size and retries automatically.

For prompt `x`, response `y`, model `theta`, and generated bias system prompt `s_b`, LLS computes:

```text
w_b(x, y) = log p_theta(y | s_b, x) - log p_theta(y | x)
```

For each preference pair with chosen response `c` and rejected response `r`, it scores:

```text
Delta_b(x, c, r) = w_b(x, c) - w_b(x, r)
```

The script keeps pairs with positive `Delta_b`, length-normalizes by the combined response length, sorts by the normalized score, and keeps the top `lls_dataset.quantile` fraction.

The selected preference dataset is saved to:

```text
{local_root}/{experiment_name}/datasets/preference_dataset.json
```

The experiment name is derived from the generated bias prompt, teacher model, truncation length, and quantile.

## Step 2: Preference Tuning

```bash
python training.py --bias dog
```

`training.py` reconstructs the same experiment path from `--bias dog`, loads `preference_dataset.json`, and trains the configured student model with DPO and LoRA.

Evaluation also follows the runtime bias. For `--bias dog`, it samples 100 completions for each prompt in `eval.gen_prompts` and counts responses containing `dog` or `dogs`. For `--bias lion`, it counts `lion` or `lions`.

Outputs are written under:

```text
{local_root}/{experiment_name}/results/{student_name}_lr{lr}_beta{beta}_rank{rank}/
```

including:

```text
progress_log.json
iterations.json
eval_samples.log
training_config.json
```

## Step 3: Inverse Logit-Linear Selection

```bash
python inverse_logit_linear_selection.py --bias dog
```

This loads the selected dataset produced by `logit_linear_selection.py --bias dog` and asks which candidate bias prompt best explains the whole preference dataset.

By default, it scores these candidates:

```text
dog cat lion tiger elephant horse dolphin eagle bear wolf
```

You can override them:

```bash
python inverse_logit_linear_selection.py --bias dog --animals dog lion cat whale fox raven horse bear tiger dolphin
```

For candidate system prompt `s`, model `M`, and selected dataset:

```text
D = {(x_i, c_i, r_i)}_{i=1}^n
```

with `c_i = r_i^+` the positive response and `r_i = r_i^-` the negative response, the headline inverse score is:

```text
Score(D; s) = (1 / n) * sum_i [
  log ( P_M(r_i^+ | s, x_i) / P_M(r_i^- | s, x_i) )
  - log ( P_M(r_i^+ | empty, x_i) / P_M(r_i^- | empty, x_i) )
]
```

Equivalently, for each pair it computes the LLS-style contrast:

```text
Delta_i(s) =
[log P_M(r_i^+ | s, x_i) - log P_M(r_i^- | s, x_i)]
- [log P_M(r_i^+ | empty, x_i) - log P_M(r_i^- | empty, x_i)]
```

and then sums `Delta_i(s)` across the dataset, up to the `1 / n` normalization.

The script ranks candidate prompts by the summed score:

```text
Score_sum(D; s) = sum_i Delta_i(s)
```

and also reports a softmax-normalized posterior-like quantity over candidate prompts:

```text
p(s | D) proportional to exp(Score_sum(D; s))
```

computed with a stable log-sum-exp normalization.

The script also reports:

```text
sum_i log P_M(r_i^+ | s, x_i)
sum_i log P_M(r_i^- | s, x_i)
sum_i log sigma(log P_M(r_i^+ | s, x_i) - log P_M(r_i^- | s, x_i))
```

These are secondary diagnostics. The main ranking now uses the contrast against the no-system-prompt baseline.

Inverse outputs are written to:

```text
{local_root}/{experiment_name}/inverse/inverse_summary.json
{local_root}/{experiment_name}/inverse/per_sample_scores.jsonl
```

## Multi-GPU / Multi-Node

The code uses Hugging Face Accelerate and extends naturally to multi-GPU and multi-node setups:

```bash
accelerate launch --num_processes <NUM_GPUS> logit_linear_selection.py --bias dog
accelerate launch --num_processes <NUM_GPUS> training.py --bias dog
```

For SLURM clusters, wrap with `srun` to ensure proper GPU allocation. See [Accelerate documentation](https://huggingface.co/docs/accelerate) for details.
