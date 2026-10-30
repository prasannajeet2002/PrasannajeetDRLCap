import csv
import math
import numpy as np

PATH = "outputs/workload_aware_100ep/evaluation_state_comparison.csv"
BASELINE = "baseline_5_feature"
AWARE = "workload_aware_11_feature"
METRICS = ["reward", "energy", "performance", "max_temp"]

# Critical t-value for a two-sided 95% CI with 29 degrees of freedom.
# This applies when there are exactly 30 paired seeds.
T_CRITICAL_29 = 2.045229642

data = {}

with open(PATH, newline="") as f:
    for row in csv.DictReader(f):
        policy = row["policy"]
        if policy not in (BASELINE, AWARE):
            continue
        seed = int(row["seed"])
        data.setdefault(seed, {})[policy] = {
            metric: float(row[metric]) for metric in METRICS
        }

seeds = sorted(
    seed for seed, policies in data.items()
    if BASELINE in policies and AWARE in policies
)

n = len(seeds)
if n < 2:
    raise SystemExit(f"Need at least 2 paired seeds; found {n}.")

print(f"Paired seeds: {n}")
print("Difference = workload-aware minus baseline")
print("Positive energy/temperature differences mean worse outcomes.\n")

results = []

for metric in METRICS:
    diffs = np.array([
        data[s][AWARE][metric] - data[s][BASELINE][metric]
        for s in seeds
    ], dtype=float)

    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / math.sqrt(n)

    # Accurate for n=30. For other sample sizes, use a conservative
    # normal critical value and clearly flag the approximation.
    if n == 30:
        critical = T_CRITICAL_29
        method = "t(29)"
    else:
        critical = 1.96
        method = "normal approximation"

    low = mean - critical * se
    high = mean + critical * se

    # Two-sided paired t-test p-value requires the t-distribution CDF.
    # We avoid reporting an invented p-value without SciPy.
    print(f"{metric}:")
    print(f"  Mean paired difference: {mean:.6f}")
    print(f"  SD of differences:      {sd:.6f}")
    print(f"  Approx. 95% CI ({method}): [{low:.6f}, {high:.6f}]")
    print(f"  CI excludes zero:       {'Yes' if low > 0 or high < 0 else 'No'}")
    print()

    results.append([metric, n, mean, sd, low, high, method])

out = "outputs/workload_aware_100ep/paired_confidence_intervals.csv"
with open(out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "metric", "n", "mean_difference", "sd_difference",
        "ci95_low", "ci95_high", "critical_value_method"
    ])
    writer.writerows(results)

print("Saved:", out)
