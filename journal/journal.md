# July 6

## Inverse Posterior Distribution

![alt text](image.png)

*Figure 1. Posterior distribution over candidate animal prompts inferred from the selected preference dataset.*

This figure shows the inverse logit-linear selection calculation over candidate latent prompts $s$. For each preference triple $(x_i, r_i^+, r_i^-)$, the script computes

$$
\Delta_i(s) =
\left[\log P_M(r_i^+ \mid s, x_i) - \log P_M(r_i^- \mid s, x_i)\right]
- \left[\log P_M(r_i^+ \mid \varnothing, x_i) - \log P_M(r_i^- \mid \varnothing, x_i)\right].
$$

It then aggregates these pairwise contrasts across the dataset:

$$
\mathrm{Score}_{\mathrm{sum}}(D; s) = \sum_{i=1}^n \Delta_i(s).
$$

Finally, the plotted quantity is the softmax-normalized posterior-like distribution over candidate prompts:

$$
p(s \mid D) = \frac{\exp\!\left(\mathrm{Score}_{\mathrm{sum}}(D; s)\right)}
{\sum_{s'} \exp\!\left(\mathrm{Score}_{\mathrm{sum}}(D; s')\right)}.
$$

Intuitively, the figure asks which animal-specific system prompt makes the observed chosen responses look most preferred relative to rejected responses, after subtracting the same preference margin under the empty system prompt baseline.

## Objective Simplification

I noted that for the purpose of choosing the maximizing latent prompt $s$, the empty-prompt baseline term is constant with respect to $s$. Since

$$
\Delta_i(s) =
\left[\log P_M(r_i^+ \mid s, x_i) - \log P_M(r_i^- \mid s, x_i)\right]
- \left[\log P_M(r_i^+ \mid \varnothing, x_i) - \log P_M(r_i^- \mid \varnothing, x_i)\right],
$$

the second bracket does not depend on $s$. Therefore,

$$
\arg\max_s \sum_i \Delta_i(s)
=
\arg\max_s \sum_i \left[\log P_M(r_i^+ \mid s, x_i) - \log P_M(r_i^- \mid s, x_i)\right].
$$

Because of this, the inverse script was simplified to rank candidate prompts using the summed system-prompt margin directly:

$$
\mathrm{Score}_{\mathrm{sum}}(D; s)
=
\sum_{i=1}^n
\left[\log P_M(r_i^+ \mid s, x_i) - \log P_M(r_i^- \mid s, x_i)\right].
$$

This change preserves the maximizing prompt while avoiding unnecessary baseline computations.

## Mean-Posterior Comparison After Metric Change

We also changed the plotting metric from the saturated posterior over the summed score to a mean-based posterior, using a softmax over `score_mean` / `inverse_score_mean`. This gives a less collapsed view of the candidate-animal distribution and makes it easier to compare runs qualitatively.

![alt text](image-1.png)

*Figure 2. Mean-posterior distribution over candidate animal prompts for the no-bias original dataset.*

## Inverse Fit With System-Prompt Embeddings

- Quantity: for each sampled system prompt `s`, the target is the summed preference margin over 500 sampled prompt-response triples.
- Fit: 500 system prompts are sampled at random from `data/system_prompts.jsonl`, with training prompts containing `dog` or `dogs` removed as a leakage check.

$$
y(s) =
\sum_i
\left[
\log P_M(r_i^+ \mid s, p_i)
-
\log P_M(r_i^- \mid s, p_i)
\right]
$$

$$
y(s) \approx a^\top e(s)
$$

**Result**
Training score: R2=0.9982, RMSE=28.7710, sign_acc=0.9935
Evaluation score: R2=-3.4203, RMSE=1190.5832, sign_acc=0.3158
Which isn't good. The problem is that I am not using that many sample points and the mean direction of the dataset isn't sampled well either. 

### Extra Details

- System-prompt embedding model: `Qwen/Qwen3-Embedding-0.6B`.
- Scoring model: the configured teacher model, currently `allenai/OLMo-2-0425-1B-Instruct`.
- The fit saves the learned vector both in a timestamped run directory and in the dog run's `inverse/` folder as `inverse_fit_a_vector.npy` / `inverse_fit_a_vector.pt`.
- The run record is appended to `inverse_fit.jsonl`, alongside training and held-out evaluation scores.

## OpenAI Embedding Cosine Probe

![Top mean cosine system prompts](../experiments/dog-lls-q0.1-trunc20/embedding_cosines/top_10_mean_cosine_sem.png)

*Figure 3. Top system prompts ranked by mean cosine similarity, with the literal dog prompt shown as an extra highlighted bar.*

![Top max cosine system prompts](../experiments/dog-lls-q0.1-trunc20/embedding_cosines/top_10_max_cosine.png)

*Figure 4. Top system prompts ranked by maximum single-example cosine similarity, again including the literal dog prompt for comparison.*

- Result: this embedding-only probe did not recover `You really love dogs.` as the top prompt.
- Mean ranking: the top prompt was `random confidence levels` with mean cosine `0.004562`; the dog prompt ranked `193` with mean cosine `0.002951`.
- Max ranking: the top prompt was `circular reasoning avoidance` with max cosine `0.260738`; the dog prompt ranked `3501` with max cosine `0.103074`.
- Quantity: completions from the dog-selected preference dataset were embedded as `User: {prompt}\nAssistant: {response}`, and prompts were embedded as `System: {system prompt}` using `text-embedding-3-large`.

$$
\mathrm{score}_{\mathrm{mean}}(s)
=
\frac{1}{n}\sum_i
\hat e(s)^\top
\left[
\hat e(p_i,r_i^+) - \hat e(p_i,r_i^-)
\right]
$$

## Extra Details

- Dataset: `14,867` preference triples from the dog LLS run.
- Candidate set: all `3,736` generated system prompts plus the literal dog prompt.
- Interpretation: simple embedding geometry did not isolate the known latent dog prompt, even though the preference dataset was selected using that target behavior.

# July 7

I wanted to test the full logprobs algorithm on the dogs prompt + another 9 random prompts. 

## Inverse Mean Posterior Replot

![Inverse mean posterior](../experiments/dog-lls-q0.1-trunc20/inverse/mean_posterior.png)

*Figure 5. Softmax over `score_mean` from the teacher-model inverse logprob run on the dog-selected dataset.*

- Result: the teacher-model inverse scoring strongly recovers the fabricated dog prompt.
- Quantity: the plot shows a softmax over candidate `score_mean` values from `experiments/dog-lls-q0.1-trunc20/inverse/summary.json`.
- Ranking: `bias:dog` receives posterior `0.924`; the next candidate, `cliche avoidance`, receives `0.025`.
- Interpretation: the actual logprob inverse method can recover the dog prompt; the embedding surrogates are failing to approximate this inverse score.

$$
p(s \mid D)
=
\frac{\exp(\mathrm{score\_mean}(s))}
{\sum_{s'} \exp(\mathrm{score\_mean}(s'))}
$$

## Combined-Length Normalization

- Change: inverse logit selection, cached logprob targets, and embedding-cosine preference directions now divide each pair score by the combined response length.
- Quantity: the logprob scripts use model-token response lengths; the OpenAI embedding cosine probe uses whitespace-token response lengths because it does not load a tokenizer.
- Interpretation: rankings now emphasize score density per response token rather than letting long `r+`/`r-` pairs dominate by accumulated raw margin.
- Auditability: raw margins are still saved alongside normalized scores as `raw_score_*`, `raw_logprob_margin`, or `raw_*_cosine` fields.

$$
\ell_i(s) =
\frac{
\log P_M(r_i^+ \mid s, p_i)
-
\log P_M(r_i^- \mid s, p_i)
}{
|r_i^+| + |r_i^-|
}
$$

For embedding cosines, the same denominator is applied to the preference direction:

$$
d_i =
\frac{
\hat e(p_i,r_i^+) - \hat e(p_i,r_i^-)
}{
|r_i^+| + |r_i^-|
}
$$

## Diagonal Matrix Reconstruction

![Top diagonal matrix system prompts](../experiments/dog-lls-q0.1-trunc20/embedding_diagonal_matrix/top_10_mean_matrix_score_sem.png)

*Figure 6. Top system prompts ranked by the mean reconstructed diagonal-matrix score on the dog-selected preference dataset, with the literal dog prompt appended in orange.*

- Result: the learned diagonal matrix did not recover `You really love dogs.` as the top prompt.
- Mean ranking: the top prompt was `random confidence levels` with mean score `0.003501`; the dog prompt ranked `1430` with mean score `0.002503`.
- Quantity: each candidate prompt is scored by applying the learned diagonal matrix `A` to the OpenAI embedding preference direction.
- Interpretation: the diagonal map changes the scale and ranking relative to plain cosine scoring, but still does not isolate the intended dog prompt.

$$
e_s = \hat e(\text{System: }s),
\quad
e_i^\pm = \hat e(\text{User: }p_i\,\text{ Assistant: }r_i^\pm)
$$

$$
\mathrm{Score}(s)
=
\frac{1}{n}
\sum_i
e_s^\top
\operatorname{diag}(A)
\frac{e_i^+ - e_i^-}{|r_i^+| + |r_i^-|}
$$

## Extra Details

- Embeddings: `s`, `(p,r+)`, and `(p,r-)` were embedded with `text-embedding-3-large` using tagged strings `System: ...` and `User: ...\nAssistant: ...`.
- Training data for `A`: `experiments/original-dataset/inverse/original_logprobs.jsonl` supplied normalized targets `logprob = raw_logprob_margin / (len(r+) + len(r-))`.
- Matrix fit: `src/fit_diagonal_logprob_matrix.py` fit a diagonal ridge model to predict the normalized logprob margin from `e(s) * (e(p,r+) - e(p,r-))`.
- Reconstruction: `src/score_preference_embedding_diagonal_matrix.py` reused the cached OpenAI embeddings for the dog-selected dataset and scored all generated system prompts plus the literal dog prompt.
- Companion max-score plot: `../experiments/dog-lls-q0.1-trunc20/embedding_diagonal_matrix/top_10_max_matrix_score.png`; the dog prompt ranked `859` by max score.

## Low-Rank Bilinear Failure Mode

![Low-rank bilinear system prompts](../experiments/original-dataset/inverse/low_rank_bilinear_openai_20260707T185558Z/top_10_mean_matrix_score_sem.png)

*Figure 7. Top system prompts ranked by the low-rank bilinear OpenAI-embedding surrogate, with the literal dog prompt appended in orange.*

- Result: the low-rank bilinear OpenAI-embedding surrogate fit the cached training task moderately well but ranked the literal dog prompt almost last on the dog-selected dataset.
- Fit quality: `Train R2=0.4954`, `Eval R2=0.4917`, with sign accuracy around `0.78` on both train and eval.
- Ranking: the top prompts were reflective/contextual traits such as `uses a reflective tone`; `You really love dogs.` ranked `3735` by mean score.
- Interpretation: this looks less like insufficient data and more like a target mismatch: the surrogate learned generic preference margin rather than the baseline-subtracted LLS effect used to fabricate the dog-selected dataset.

The current bilinear target is approximately:

$$
y(s,p,r^+,r^-)
=
\frac{
\log P_M(r^+ \mid s,p)
-
\log P_M(r^- \mid s,p)
}{
|r^+|+|r^-|
}
$$

But the forward LLS selection signal was a system-prompt effect relative to the empty prompt:

$$
\Delta_s
=
\frac{
\left[
\log P_M(r^+ \mid s,p)-\log P_M(r^+ \mid \varnothing,p)
\right]
-
\left[
\log P_M(r^- \mid s,p)-\log P_M(r^- \mid \varnothing,p)
\right]
}{
|r^+|+|r^-|
}
$$

## Extra Details

- The failure is not circular evidence against inverse recovery: the dog dataset was fabricated by forward LLS, so a non-dog-trained inverse method should recover the dog prompt if it approximates the same score.
- The next target should be `margin_s - margin_empty`, not just `margin_s`.
- This still trains on random system prompts and original preference pairs; it only changes the supervised target to match the forward LLS construction.
