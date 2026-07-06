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

On 2026-07-06, we noted that for the purpose of choosing the maximizing latent prompt $s$, the empty-prompt baseline term is constant with respect to $s$. Since

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

One possible reason is that the candidate prompts are all very similar in form, since each one follows essentially the same template of “you really love `<animal>`.” That may limit how separable the latent prompts are, even when the underlying dataset is biased.
