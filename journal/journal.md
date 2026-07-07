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

# July 7

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
- Fit: 500 system prompts are sampled at random from `runs/system_prompts/system_prompts.jsonl`, with training prompts containing `dog` or `dogs` removed as a leakage check.

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

![Top mean cosine system prompts](../runs/You_really_love_dogs_Dogs_are_8b18099e_OLMo-2-0425-1B-Instruct_trunc20_q0.1/embedding_cosines/top_10_mean_cosine_sem.png)

*Figure 3. Top system prompts ranked by mean cosine similarity, with the literal dog prompt shown as an extra highlighted bar.*

![Top max cosine system prompts](../runs/You_really_love_dogs_Dogs_are_8b18099e_OLMo-2-0425-1B-Instruct_trunc20_q0.1/embedding_cosines/top_10_max_cosine.png)

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
