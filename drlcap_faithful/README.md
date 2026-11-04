# drlcap_faithful: paper-aligned DRLCap on the RTX 4070 Ti

Started 2026-11-04 after the code review "DRLCAP code review - changes needed
before retraining". `drlcap_repro/` (v1-v4) and all its results are kept
unchanged; this folder starts again from the paper.

Nothing has been trained here yet. The reviewer asked to see four things
first; they are:

| Review request | File |
|---|---|
| Explanation of the v4 changes (items 1-2) | `V4_CHANGES.md` |
| Transition-timing check with a timestamped trace (item 3) | `TRACE_CHECK.md`, `results/trace_gaussian_delay{0,1}.csv` |
| Proposed paper-aligned configuration (items 1-2) | `REFERENCE_CONFIG.md` |
| Expanded benchmark table (item 5) | `BENCHMARKS.md`, `results/benchmark_table.csv` |

## Files

| File | Purpose |
|---|---|
| `drlcap.py` | controller + offline training (copied from drlcap_repro, set to the paper; `trace` command; one-setting variant flags) |
| `agent.py`, `reward.py`, `gpu_control.py`, `rl_baseline.py`, `evaluate.py` | copied from drlcap_repro; agent γ = 0.95, reward mode `alg1` |
| `workloads.py` | 64 training, 11 validation and 10 test programs (paper suites and Table II) |
| `cnn_workload.py` | VGG-16 and GoogLeNet trained on CIFAR-100 |
| `bench_probe.py` | runs programs on GPU 1, checks GPU placement, records time / util / power |
| `bench_table.py`, `make_benchmarks_md.py` | build the benchmark table |
| `../benchmarks/build_scripts/` | how every suite was built (logs in `../benchmarks/build_logs`, not in git) |

## Commands

```
python bench_probe.py [--job] [names...]       # run programs once (or a whole job)
python drlcap.py trace gaussian --out t.csv    # timestamped transition trace
python drlcap.py train --tag ref_s0 --seed 0   # reference training (NOT run yet)
```
