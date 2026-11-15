"""Choose the DRLCap model for the final test on the VALIDATION programs.

The trained policy depends on the random seed (f1: 10.8 / 14.2 / 11.0 %
EDP for seeds 0 / 1 / 2 on the test programs). The paper does not say how
its final model was chosen, only that settings were fixed to "the
best-performing value found during offline training" (Sec. IV-D). We
train several seeds and keep the one with the best geometric-mean EDP on
the validation programs (workloads.VAL: held out of training, never part
of the test set). The test programs are not looked at for this choice.

    python select_model.py results/eval_val_f2_<stamp> [--out selected.json]

Prints the chosen model path on the last line.
"""

import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    with open(os.path.join(args.eval_dir, "summary.csv")) as f:
        geo = [r for r in csv.DictReader(f) if r["workload"] == "GEOMEAN"]
    with open(os.path.join(args.eval_dir, "config.json")) as f:
        cfg = json.load(f)
    table = [{"policy": r["policy"],
              "model": cfg["models"][r["policy"]][0],
              "energy_saving_pct": round(float(r["energy_saving_pct"]), 2),
              "perf_loss_pct": round(float(r["perf_loss_pct"]), 2),
              "edp_improvement_pct": round(float(r["edp_improvement_pct"]), 2)}
             for r in geo]
    best = max(table, key=lambda r: r["edp_improvement_pct"])
    out = args.out or os.path.join(args.eval_dir, "selected.json")
    with open(out, "w") as f:
        json.dump({"criterion": "best geometric-mean EDP improvement on the "
                                "validation programs",
                   "validation_programs": cfg["workloads"],
                   "repeats": cfg["repeats"], "candidates": table,
                   "selected": best}, f, indent=2)
    for r in table:
        mark = "  <- selected" if r is best else ""
        print(f"{r['policy']:10s} energy {r['energy_saving_pct']:6.2f}%  "
              f"perf loss {r['perf_loss_pct']:6.2f}%  "
              f"EDP {r['edp_improvement_pct']:6.2f}%{mark}")
    print(best["model"])


if __name__ == "__main__":
    main()
