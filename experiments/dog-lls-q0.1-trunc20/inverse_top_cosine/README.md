# Inverse Top-Cosine Prompt Check

This folder is for the follow-up inverse logit-linear selection run that compares the dog-bias prompt against the highest-scoring system prompts from the embedding-cosine analysis.

The goal is to check whether the dog prompt still gets high inverse posterior probability when the comparison set is not random prompts, but prompts that already looked most similar to the dog-selected preference direction in Figure 3.

## How The Candidates Were Chosen

The candidate list is created by:

```powershell
python src/inverse_top_cosine_system_prompts.py
```

That script reads:

```text
experiments/dog-lls-q0.1-trunc20/embedding_cosines/system_prompt_cosines.summary.json
```

and extracts rows from its `scores` array, sorted by `rank`.

If that summary file is not present locally, the script first tries to pull `data/` and `experiments/` from the Hugging Face dataset configured in `config.yaml`.

The current candidate set contains 10 prompts:

1. The explicit dog-bias prompt from `helper_functions.bias_system_prompt("dog")`.
2. The top 9 non-literal system prompts by embedding-cosine rank from `system_prompt_cosines.summary.json`.

The wrapper skips rows whose `source` is `literal`, then keeps the first 9 unique prompts. In this run, those are embedding-cosine ranks 1 through 9.

## Files

### `candidate_prompts.jsonl`

Input file for `src/inverse_logit_linear_selection.py`.

Each JSONL row is one candidate system prompt to score against the dog-selected preference dataset. The first row is the dog-bias prompt. The remaining rows were extracted from `system_prompt_cosines.summary.json` and preserve the useful provenance fields:

- `embedding_cosine_rank`: rank from the embedding-cosine summary.
- `mean_cosine`: the Figure 3 ranking score.
- `source_index`: original index in the system prompt source list.
- `category`, `trait`, `trait_normalized`: metadata from the system prompt dataset.
- `system_prompt`: the actual candidate prompt used for inverse scoring.

### `metadata.json`

Created after running the script without `--dry-run`.

Records the inverse run configuration, including the source bias, dataset path, model, posterior metric, candidate source label, and the full candidate list that was scored.

### `candidate_scores.jsonl`

Created after running the script without `--dry-run`.

Append-only running output with one row per scored candidate. This is useful for checking partial progress during a long run. Each row includes aggregate inverse scores such as:

- `score_sum`
- `score_mean`
- `raw_score_sum`
- `preference_logprob_sum`

The final posterior-style normalization is added in `summary.json`.

### `per_sample_scores.jsonl`

Created after running the script without `--dry-run`.

One row per preference example. Each row contains the prompt, chosen response, rejected response, and per-candidate scoring details, including chosen/rejected logprobs, response lengths, raw margins, normalized margins, and preference logprob.

### `summary.json`

Created after running the script without `--dry-run`.

Final ranked inverse result. This is the main file to inspect after the run. It sorts candidates by `score_sum` and includes `posterior_from_score`, which is the posterior-like normalized probability used to compare the dog prompt against the top-cosine alternatives.

## Reproducing Just The Candidate Extraction

To regenerate `candidate_prompts.jsonl` without launching the model-scoring run:

```powershell
python src/inverse_top_cosine_system_prompts.py --dry-run
```

To run the full inverse comparison:

```powershell
python src/inverse_top_cosine_system_prompts.py
```
