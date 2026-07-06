## Posterior Plots

Create a bar chart where each animal is a bar and the bar height is the selected metric value:

```bash
python plots/plot_posterior_distribution.py path/to/inverse_summary.json
```

If no path is provided, the script looks for the most recent `runs/*/inverse/inverse_summary.json`.

By default, the script plots `mean_posterior`, which is a softmax over `inverse_score_mean`.

Other useful posterior-style views:

- `--metric sum_posterior`
- `--metric stored_posterior`

You can also plot any numeric field already saved in each animal row, for example:

```bash
python plots/plot_posterior_distribution.py path/to/inverse_summary.json --metric inverse_score_sum
python plots/plot_posterior_distribution.py path/to/inverse_summary.json --metric chosen_logprob_per_token
python plots/plot_posterior_distribution.py path/to/inverse_summary.json --metric num_examples
```

Optional output path:

```bash
python plots/plot_posterior_distribution.py path/to/inverse_summary.json --output plots/posterior_distribution.png
```
