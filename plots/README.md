## Posterior Plots

Create a bar chart where each animal is a bar and the bar height is that animal's posterior probability:

```bash
python plots/plot_posterior_distribution.py path/to/inverse_summary.json
```

If no path is provided, the script looks for the most recent `runs/*/inverse/inverse_summary.json`.

Optional output path:

```bash
python plots/plot_posterior_distribution.py path/to/inverse_summary.json --output plots/posterior_distribution.png
```
