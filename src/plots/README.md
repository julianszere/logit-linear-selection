## Posterior Plots

Create a bar chart where each animal is a bar and the bar height is the selected metric value:

```bash
python src/plots/plot_posterior_distribution/plot_posterior_distribution.py experiments/dog-lls-q0.1-trunc20/inverse/summary.json
```

If no path is provided, the script looks for the most recent `experiments/*/inverse/summary.json`.

By default, the script plots `mean_posterior`, which is a softmax over `inverse_score_mean`.

Other useful posterior-style views:

- `--metric sum_posterior`
- `--metric stored_posterior`

You can also plot any numeric field already saved in each animal row, for example:

```bash
python src/plots/plot_posterior_distribution/plot_posterior_distribution.py experiments/dog-lls-q0.1-trunc20/inverse/summary.json --metric score_sum
python src/plots/plot_posterior_distribution/plot_posterior_distribution.py experiments/dog-lls-q0.1-trunc20/inverse/summary.json --metric preference_logprob_sum
python src/plots/plot_posterior_distribution/plot_posterior_distribution.py experiments/dog-lls-q0.1-trunc20/inverse/summary.json --metric num_examples
```

Optional output path:

```bash
python src/plots/plot_posterior_distribution/plot_posterior_distribution.py experiments/dog-lls-q0.1-trunc20/inverse/summary.json --output experiments/dog-lls-q0.1-trunc20/inverse/posterior.png
```
