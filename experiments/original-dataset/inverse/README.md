# Original-Dataset Inverse Experiments

This folder stores inverse-scoring artifacts for the original preference dataset and several embedding-based fits that try to approximate the normalized log-probability margins.

The core target is

```text
logprob = log P_M(r_plus | s, p) - log P_M(r_minus | s, p)
```

where `M` is `allenai/OLMo-2-0425-1B-Instruct`, `s` is a system prompt, `p` is a user prompt, and `r_plus`, `r_minus` are the preferred and rejected responses.

## Data Generation

`original_logprobs.jsonl` is the main scored dataset in this folder.

It was generated from:

```text
dataset: data/original_preferences.json
system prompts: data/system_prompts.jsonl
model: allenai/OLMo-2-0425-1B-Instruct
seed: 0
```

The run scored:

- 300 system prompts
- 500 preference pairs per system prompt
- 15,000 available original preference pairs
- 150,000 total scored rows

Each row contains the paired fields `s`, `p`, `r_plus`, `r_minus`, `chosen_logprob`, `rejected_logprob`, `logprob`, `raw_logprob_margin`, and `length_denominator`.

`original_logprobs.summary.json` records the generation metadata.

## Embeddings

`original_logprob_embeddings.npz` contains the OpenAI embeddings used by the fit scripts.

The embeddings were produced by `src/embed_original_logprob_rows.py` with:

```text
embedding model: text-embedding-3-large
storage dtype: float32
format: compact_unique_embeddings
```

Arrays:

- `unique_embeddings`: shape `(30290, 3072)`
- `unique_texts`: unique formatted texts
- `system_text_indices`: shape `(150000,)`
- `chosen_text_indices`: shape `(150000,)`
- `rejected_text_indices`: shape `(150000,)`
- `target_logprob_margins`: shape `(150000,)`
- `raw_logprob_margins`: shape `(150000,)`
- `length_denominators`: shape `(150000,)`

Text formatting:

```text
system:   System: {s}
response: User: {p}\nAssistant: {r}
```

`original_logprob_embedding_cache.text-embedding-3-large.npz` is the per-text cache used to build the compact embedding file.

## Existing Animal Inverse Run

The root-level `summary.json`, `candidate_scores.jsonl`, and `per_sample_scores.jsonl` are from an animal-prompt inverse comparison on the original dataset.

In that run, `fox` had the highest posterior-like score and the dog prompt ranked below several other animal prompts. The scored candidate rows include aggregate fields such as `inverse_score_sum`, `inverse_score_mean`, logprob totals, per-token logprob totals, and `posterior_from_inverse_score`.

## Diagonal Fit

Folder:

```text
diagonal_fit_20260707T183559Z/
```

Script:

```text
src/fit_diagonal_logprob_matrix.py
```

Model:

```text
y = e(s)^T diag(A) (e(p,r_plus) - e(p,r_minus))
```

This fit used the precomputed `text-embedding-3-large` embeddings, ridge `0.001`, seed `0`, and an 80/20 train/held-out split.

Results:

| split | rows | RMSE | R2 | sign accuracy |
| --- | ---: | ---: | ---: | ---: |
| train | 120,000 | 0.6565 | 0.7864 | 0.8522 |
| heldout | 30,000 | 0.6607 | 0.7812 | 0.8482 |

Important files:

- `metrics.json`: training configuration and metrics
- `A_diagonal.npy`: learned diagonal weights
- `A_matrix.npy`: dense diagonal matrix
- `A_matrix.pt`: PyTorch checkpoint copy
- `predictions.jsonl`: per-row predictions and residuals
- `train_predictions.npy`, `eval_predictions.npy`
- `train_targets.npy`, `eval_targets.npy`

## PCA Low-Rank Bilinear Fit

Folder:

```text
low_rank_bilinear_openai_20260707T185558Z/
```

Script:

```text
src/fit_and_score_low_rank_bilinear.py
```

Model:

```text
score = PCA(e_s)^T B ((PCA(e_plus) - PCA(e_minus)) / length)
```

This fit uses a rank-64 PCA basis over the cached embeddings and fits a rank-space bilinear matrix with ridge `0.001`.

Results:

| split | rows | RMSE | R2 | sign accuracy |
| --- | ---: | ---: | ---: | ---: |
| train | 120,000 | 1.0089 | 0.4954 | 0.7813 |
| eval | 30,000 | 1.0069 | 0.4917 | 0.7792 |

Dog-prompt scoring on `data/dog_selected_preferences.json`:

```text
rank: 3735 / 3737
mean_matrix_score: 0.1211641124
max_matrix_score_rank: 3735
num_examples: 14867
```

Top mean-scoring prompts in this run were mostly reflective/context-sensitive prompts.

Important files:

- `summary.json`: configuration, metrics, dog prompt result, and top scored prompts
- `system_prompt_low_rank_bilinear_scores.jsonl`: all scored system prompts
- `B_low_rank.npy`: learned rank-space bilinear matrix
- `pca_mean.npy`, `pca_components.npy`: PCA transform
- `top_10_mean_matrix_score_sem.png`: plot of top scores
- `train_predictions.npy`, `eval_predictions.npy`
- `train_targets.npy`, `eval_targets.npy`

## SVD-Factor One-Layer MLP Fit

Script:

```text
src/fit_and_score_svd_mlp_bilinear.py
```

This script trains the two one-layer maps:

```text
psi(s) = W_s e_s(s)
phi(p,r) = W_pr e_pr(p,r)
```

It first reconstructs the observed system-by-preference-pair matrix from `original_logprobs.jsonl`.

The intended matrix shape is:

```text
300 system prompts x 15000 preference pairs
```

Only 150,000 cells were observed, because each system prompt was scored on 500 sampled preference pairs. The script therefore builds the train-only observed matrix, fills missing cells before SVD, computes a rank-k factorization,

```text
M_k = U_k Sigma_k V_k^T
Z_s = U_k Sigma_k^0.5
Z_pr = V_k Sigma_k^0.5
```

and then fits:

```text
W_s e_s(s_i) ~= Z_s[i]
W_pr e_pr(p_j,r_j) ~= Z_pr[j]
```

For this preference-margin setup, the prompt-response embedding is:

```text
e_pr(p_j,r_j) = (e(User: p_j\nAssistant: r_plus_j) - e(User: p_j\nAssistant: r_minus_j)) / length_denominator_j
```

Run with the default `k = rank(M)` behavior:

```powershell
python src/fit_and_score_svd_mlp_bilinear.py --ridge 1e-3
```

Here `rank(M)` is the numerical rank of the filled training matrix after applying the missing-cell fill rule. You can still override `k` manually for ablations:

```powershell
python src/fit_and_score_svd_mlp_bilinear.py --rank 64 --ridge 1e-3
```

Default output folders are named:

```text
svd_mlp_bilinear_openai_<timestamp>/
```

Expected outputs:

- `summary.json`: fit configuration, metrics, dog prompt result, and top scored prompts
- `system_prompt_svd_mlp_bilinear_scores.jsonl`: all scored system prompts
- `W_system.npy`: learned `W_s`
- `W_preference.npy`: learned `W_pr`
- `svd_singular_values.npy`: singular values from the filled training matrix
- `train_predictions.npy`, `eval_predictions.npy`
- `train_targets.npy`, `eval_targets.npy`

Default missing-cell fill is `column_mean`. Other options:

```powershell
python src/fit_and_score_svd_mlp_bilinear.py --fill-value global_mean
python src/fit_and_score_svd_mlp_bilinear.py --fill-value zero
```

## Reproduction Order

To rebuild this folder from source data:

1. Score original preference pairs under sampled system prompts to produce `original_logprobs.jsonl`.
2. Embed the scored rows:

```powershell
python src/embed_original_logprob_rows.py
```

3. Run the diagonal fit:

```powershell
python src/fit_diagonal_logprob_matrix.py --embeddings-path experiments/original-dataset/inverse/original_logprob_embeddings.npz
```

4. Run the PCA low-rank bilinear fit:

```powershell
python src/fit_and_score_low_rank_bilinear.py
```

5. Run the SVD-factor one-layer MLP fit:

```powershell
python src/fit_and_score_svd_mlp_bilinear.py --ridge 1e-3
```

The fit scripts use `config.yaml` to pull and push `data/` and `experiments/` through the configured Hugging Face dataset when HF sync is enabled.
