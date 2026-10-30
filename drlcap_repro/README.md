# DRLCap reproduction on the RTX 4070 Ti

Reproduction of *DRLCap: Runtime GPU Frequency Capping With Deep
Reinforcement Learning* (Wang et al., IEEE TSUSC 9(5), 2024) on an NVIDIA
GeForce RTX 4070 Ti (Ada, physical GPU index 1). GPU 0 (RTX 4090) is never
used.

Run everything inside the project environment:

```bash
source /home/drl/miniconda3/etc/profile.d/conda.sh
conda activate prasannajeet_env
cd /home/drl/project/drlcap/drlcap_repro
```

## Files

| File | Paper component |
|---|---|
| `workloads.py` | Training / testing benchmarks (Sec. V-B, Table II) |
| `gpu_control.py` | Workload monitor + power controller (Fig. 4): NVML telemetry, `nvidia-smi -lgc` frequency cap, always reset on exit |
| `frequency_sweep.py` | Motivation experiment (Sec. III, Fig. 1) and action-range selection (Table IV) |
| `reward.py` | Energy-efficiency evaluator: heuristic reward, Algorithm 1 + Eq. 1 |
| `agent.py` | Power-configuration optimizer: DDQN + prioritized replay (Alg. 2, Fig. 7-8, Table V) |
| `drlcap.py` | 300 ms runtime control loop, discretised state, frequency / power / hybrid variants (Table III); `python drlcap.py train` = offline training |
| `rl_baseline.py` | The paper's "RL" baseline: tabular double Q-learning, same actions and reward |
| `evaluate.py` | Comparison with the NVIDIA default: static caps, COORD-like heuristic, trained models, monitor-only overhead; energy, runtime, EDP, geomean (Sec. V-D, VI-A..C) |
| `cnn_workload.py` | VGG-16 / GoogLeNet (InceptionV3) / ResNet50 training test workloads |
| `report.py` | Figures and Markdown table from an evaluation's `summary.csv` |
| `run_full_reproduction.sh` | The whole queue: 5 trainings, main comparison, ablations |

Results go to `results/` with timestamped names; nothing is overwritten.

## Commands

```bash
# 1. static cap sweep (motivation + action range)
python frequency_sweep.py --workloads hotspot gaussian nw bfs particlefilter

# 2. offline training on the training set (5000 iterations, Table V)
python drlcap.py train

# 3. test on unseen workloads vs. NVIDIA default, online learning on
python evaluate.py --models drlcap=results/train_<tag>_<stamp>/model_final.keras \
    --repeats 10 --static-caps 2100 2400 --coord

# everything (trainings + comparison + ablations, ~14 h on GPU 1)
./run_full_reproduction.sh
```

## Where the unpublished constants come from

The paper does not publish r1..r4, g1..g4, n, c, the hidden-layer width,
the optimizer or the PER details. The project root contains a TF2 port of
the authors' own V100S scripts (`Freq_DDQN_prioritized_save_tf2.py`,
`*_restore_tf2.py`, `Power_DDQN_prioritized_restore_tf2.py`; paths
`/home/wym/...`, `nvidia-smi -lgc 135,<cap>`). From them we take: Eq. 1
bins of 45 MHz with g = +5/-1/-2/-3 (power not up) and -1/-2/-3/-4 (power
up), power floored to whole watts; 20 hidden ReLU units, N(0, 0.3)
weights, 0.1 bias, RMSprop; PER alpha 0.6, priority epsilon 0.01,
priorities clipped at 1, beta 0.4 + 0.001 per batch, new transitions at
the current maximum priority; learning only after the buffer is full;
a bucketed state; deployment = restore + online learning at fixed
epsilon. Those scripts use lr 1e-4, epsilon 0.3 -> 0.2, buffer 500,
target every 10, 2000 steps (apparently an earlier version); where they
conflict with the paper we follow Table V. Algorithm 1 cases A-D are not
in the scripts, so r1..r4, p1..p3 and c remain our choices.

## Static frequency-cap sweep (Fig. 1 analogue)

`results/frequency_sweep_20261030_113800.csv`, one run per point, change
relative to the NVIDIA default (time / energy / EDP, %):

| Cap (MHz) | hotspot | gaussian | particlefilter | nw | bfs |
|---|---|---|---|---|---|
| 1200 | +118 / +1 / +120 | +78 / -16 / +50 | +124 / +4 / +133 | +1 / -38 / -37 | +4 / -27 / -24 |
| 1800 | +48 / -22 / +15 | +32 / -28 / -4 | +50 / -21 / +19 | 0 / -37 / -37 | +4 / -23 / -21 |
| 2100 | +28 / -27 / -6 | +19 / -30 / -17 | +29 / -26 / -4 | 0 / -35 / -35 | +1 / -22 / -21 |
| 2400 | +13 / -21 / -11 | +9 / -25 / -18 | +14 / -21 / -10 | +1 / -27 / -26 | +2 / -12 / -10 |
| 2700 | +1 / -1 / 0 | 0 / -5 / -5 | +1 / -2 / -1 | +1 / -10 / -8 | +1 / +1 / +2 |

Default clocks under load: 2625-2745 MHz. Compute-bound programs have an
energy minimum near 2100 MHz at a large slowdown; memory/host-bound ones
save 22-37 % energy at no slowdown. As on the V100S (paper Fig. 1), the
best static cap differs per program, which is the motivation for a
dynamic policy. The action range 1800-2745 MHz (step 15, 64 actions)
follows from this sweep.

Observed measurement detail: NVML clock/power readings reflect a new cap
about one 300 ms sample late, so the reward of action a_t is partly
computed on the effect of a_(t-1). The paper does not discuss this.

## Results v1 (our own constants, superseded)

These runs used OUR guesses for the unpublished constants (64 hidden
units, Adam, g in [-1, 1], 30 MHz bins, continuous state; code at commit
d551cdc). They are kept because they are a useful ablation of the
constants.

Training: `results/train_drlcap_20261030_120328` - 5014 control steps
(300 ms) over 55 jobs of the 8 training workloads, 4982 DDQN updates,
epsilon 0.3 -> 0.1. Algorithm 1 cases seen: B 1694, A 1363, D 1037, C 920.

Test on the 7 unseen workloads: `results/eval_test_20261030_122906`,
3 repeats per policy, alternating order, online learning on,
epsilon 0.1. Energy from the NVML counter.

| Workload | Energy saving | Perf. loss | EDP improvement | Baseline energy CV |
|---|---|---|---|---|
| b+tree | 8.8 % | 0.6 % | 8.3 % | 1.2 % |
| dwt2d | 8.3 % | 2.5 % | 5.9 % | 1.3 % |
| heartwall | 19.2 % | 1.5 % | 18.0 % | 0.2 % |
| lavaMD | 5.4 % | 0.7 % | 4.8 % | 0.3 % |
| lud | 20.2 % | 9.3 % | 12.8 % | 0.1 % |
| myocyte | 15.5 % | 3.7 % | 12.4 % | 1.0 % |
| streamcluster | 25.0 % | -8.3 % * | 31.2 % | 3.9 % |
| **Geomean** | **14.9 %** | **1.3 %** | **13.8 %** | |

For every workload, all three DRLCap runs used less energy than all three
baseline runs. Paper (NVIDIA average): 22 % / 3 % / 20 %.

\* streamcluster's baseline runtime varies by 7 % (31.9-37.3 s), so its
apparent speed-up is within noise and should not be claimed.

10 repeats with static caps (`results/eval_test10_20261030_151611`,
same model, same 7 programs), geomean vs. NVIDIA default:

| Policy | Energy saving | Perf. loss | EDP improvement |
|---|---|---|---|
| static 2100 MHz | 26.3 % | 10.2 % | 18.8 % |
| static 2400 MHz | 19.8 % | 4.3 % | 16.3 % |
| DRLCap v1 | 15.1 % | 3.8 % | 11.9 % |

With 10 repeats the v1 slowdown is 3.8 % (3 repeats suggested 1.3 %),
and a single fixed 2400 MHz cap beats v1 on energy and EDP at a similar
slowdown. v1 therefore does NOT show an advantage of the learned policy
over the best static setting.

## What matches the paper

- State: GPU core utilisation, memory utilisation, power, SM frequency,
  temperature (Sec. IV-B), sampled every 300 ms (Sec. VI-C).
- Action: GPU core frequency cap (the paper's main "Frequency" variant).
- Reward: Algorithm 1 cases A-D with the Eq. 1 piecewise frequency score.
- Agent: DDQN, prioritized replay in a sum-tree with max-priority insertion,
  importance-sampling weights, |TD| priority updates; one-hidden-layer
  FCNN; target network copied every 8 updates.
- Table V hyper-parameters: batch 32, lr 0.001, buffer 512, gamma 0.95,
  epsilon 0.3 -> 0.1, 5000 training iterations, target update 8.
- Offline training on training benchmarks, then deployment on unseen
  benchmarks with online learning continuing (Sec. IV-D).
- Agent runs as a CPU process; the application is not modified.
- Baseline is the NVIDIA default clock management; metrics are energy,
  runtime and EDP; geometric mean across workloads.

## Deviations (must be stated in the thesis)

1. **Hardware**: RTX 4070 Ti (Ada) instead of V100S / 3080 Ti / 2080 Ti.
2. **Benchmarks**: only Rodinia 3.1 is available here. 17 of 22 CUDA
   programs build with CUDA 12.3 (cfd, kmeans, hybridsort, srad,
   hotspot3D do not; leukocyte needs CUDA 5.5 runtime). Rodinia is split
   into a training and a test set, instead of training on
   Rodinia/Parboil/SHOC/PolyBench/SDK and testing on LULESH, CloverLeaf,
   etc. Inputs are enlarged and programs repeated so a job lasts ~20-50 s.
3. **CNN workloads**: GoogLeNet is replaced by Keras InceptionV3 (same
   family; Keras has no Inception v1); data are synthetic. TensorFlow uses
   the CUDA 12.9 / cuDNN 9.26 libraries of the base miniconda install via
   LD_LIBRARY_PATH, because the pip CUDA libraries in `prasannajeet_env`
   are CUDA 13 builds that driver 575 cannot run. QE, LULESH, CloverLeaf,
   TeaLeaf, miniBUDE, CoMD, GROMACS and miniFE are not available.
4. **Unpublished constants**: taken from the authors' scripts where
   possible (section above); r1..r4, p1..p3, c, the state bucket ranges
   for the 4070 Ti and the action ranges are ours. All values are saved
   in each run's `config.json`. The state bucket indices are scaled to
   [0, 1) (the scripts feed raw indices).
5. **Reward clipping** to +-10 (case C divides by utilisation).
6. **Case A** is implemented by forcing the next action to the maximum cap
   when the latest sample is fully compute-bound (u_core = 100,
   u_mem = 0); the reward for that step then bins the power change.
7. **Energy** is the NVML board energy counter (joules), not
   power x time; the paper does not state its energy source.
8. **Repeats**: the main comparison uses 10 repeats as in the paper; the
   ablations use 5.
9. **Power capping on the 4070 Ti** accepts 100-285 W, but the Rodinia
   programs draw only ~30-100 W, so power caps rarely bind on them; they
   do bind on the CNN workloads.
10. **COORD** is a COORD-like re-implementation: one profiling run at
    default clocks, compute- vs. memory/host-bound by mean core
    utilisation (threshold 50 %), fixed caps 2400 / 1800 MHz chosen from
    the training-set sweep. The 4070 Ti has no memory-power cap, so the
    core/memory coordination of the original is not reproduced.
11. **Overhead**: CPU package energy (paper Fig. 12) needs root access to
    RAPL and is not measured; we report the controller's CPU time,
    per-step latency and memory instead, and the GPU energy/time effect
    of running the loop without applying caps (Fig. 11 analogue).
12. **Online learning** in deployment starts after one batch (32 samples),
    since a test job is shorter than the 512 steps needed to fill the
    buffer; offline training waits for a full buffer as the authors do.
