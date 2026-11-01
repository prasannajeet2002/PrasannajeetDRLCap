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
| `analyze_eval.py` | Rodinia-only / CNN-only geomeans and bootstrap 95 % intervals from `runs.csv` |
| `probe_policy.py` | Greedy action of a trained Q-network over 2000 random states (is the policy state-dependent?) |
| `diag_monitor_effect.py` | Diagnostic for the monitor-only energy difference (baseline vs. NVML polling vs. monitor-only) |
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

## Results v2: full reproduction (authors' constants), 2026-10-31

Queue `run_full_reproduction.sh`, log `results/queue_20261030_180321.log`,
2026-10-30 18:03 - 2026-10-31 09:36, no failed job. Ten unseen test
programs (7 Rodinia + VGG-16, GoogLeNet/InceptionV3, ResNet50), every
policy run interleaved with rotating order. All numbers are relative to
the NVIDIA default measured in the same evaluation; energy = NVML board
energy counter. Test results only; training statistics are in each
`results/train_*/jobs.csv`.

### Training (5000 iterations each)

`train_freq_s0_20261030_180323`, `train_freq_s1_20261030_182848`,
`train_power_s0_20261030_185400`, `train_hybrid_s0_20261030_191924`,
`train_rl_s0_20261030_194438` (tabular). freq_s0: loss median per
quarter 34 -> 172 -> 101 -> 52, mean reward flat at ~3.5, Algorithm 1
cases B 1688 / A 1358 / D 1039 / C 965. A flat mean reward means the
training curve gives no evidence of learning by itself.

### Main comparison, 10 repeats (`results/eval_main_20261030_200956`)

Geomean over the 10 test programs (`table_geomean.md`, `analysis.md`,
`fig_policies.png`, `fig_per_workload_<policy>.png`):

| Policy | Energy saving | Perf. loss | EDP improvement | EDP 95 % CI |
|---|---|---|---|---|
| static 2100 MHz | 28.4 % | 7.9 % | 22.7 % | 21.9 – 23.6 |
| static 2400 MHz | 21.0 % | 3.5 % | 18.3 % | 17.5 – 19.1 |
| COORD-like | 27.5 % | 6.5 % | 22.8 % | 21.9 – 23.6 |
| **DRLCap (frequency)** | **21.9 %** | **4.3 %** | **18.5 %** | 17.6 – 19.5 |
| DRLCap power | 7.9 % | 2.8 % | 5.3 % | 4.3 – 6.4 |
| DRLCap hybrid | 14.3 % | 3.2 % | 11.6 % | 10.6 – 12.6 |
| RL (tabular double Q) | 21.7 % | 4.6 % | 18.1 % | 17.2 – 18.9 |

The CI is a bootstrap over repeats (`analyze_eval.py`). Paper, DRLCap vs.
NVIDIA default: 22 % / 3 % / 20 % on average, 27 % / 2.4 % / 23 % on the
V100S; paper Table III also has COORD 25 / 5 / 21 and RL 5 / 1 / 4.

DRLCap per program (energy / perf. loss / EDP, %): b+tree 22.3 / 1.7 /
21.0, dwt2d 18.3 / 5.6 / 13.8, heartwall 26.7 / 2.0 / 25.2, lavaMD 6.6 /
1.7 / 5.0, lud 21.2 / 10.7 / 12.9, myocyte 27.2 / 14.9 / 16.3,
streamcluster 25.4 / -1.9 / 26.9 (baseline time CV 6.6 %, within noise),
vgg16 22.2 / 3.6 / 19.5, googlenet 22.7 / 3.2 / 20.2, resnet50 24.0 /
2.0 / 22.5. Rodinia-only geomean 21.4 / 4.8 / 17.6, CNN-only 23.0 / 3.0 /
20.7.

Facts:
- DRLCap's headline numbers are close to the paper's (21.9 / 4.3 / 18.5
  vs. 22 / 3 / 20).
- But DRLCap is statistically tied with one fixed 2400 MHz cap and is
  ~4 points EDP worse than a fixed 2100 MHz cap and the COORD-like
  heuristic. The paper's claim that DRLCap beats COORD and RL is NOT
  reproduced here; tabular RL is as good as DRLCap (paper: RL 4 % EDP).
- The power variant barely acts on Rodinia (2-3 % saving; those
  programs draw less than the 100 W minimum power limit, deviation 9);
  on the CNNs it saves ~16 %. The hybrid variant lies in between.
- The COORD-like profiling classified all three CNNs and heartwall as
  memory/host-bound (mean util 31-47 %), so they ran at 1800 MHz.

### Why: the trained policies are (almost) constant

`probe_policy.py` (`results/probe_policy_20261031.txt`) feeds 2000 random
states to each saved network:

| Model | Greedy action over 2000 random states |
|---|---|
| freq_s0 | 2160 MHz in 99.8 % |
| freq_s1 | 2100 MHz in 99.3 % |
| power_s0 | 140 W 56.5 %, 135 W 42.7 % |
| hybrid_s0 | 2610 MHz / 153 W in 96.8 % |
| v1 model (our old constants) | 2700 MHz 68.5 %, 2355 MHz 23.4 % |

The Q-values do depend on the state (they rise and fall with it), but the
ORDER of the actions hardly does: the median gap between the best and the
second-best action is 0.2-2 while Q spans 40-75. In the test runs the
same holds: in the frozen ablation every non-random, non-forced step chose
2160 MHz. The only per-program behaviour comes from Algorithm 1 case A
(forced maximum cap when util = 100 %, mem = 0 %), which is why lavaMD
runs at ~2620 MHz and saves only 6.6 % energy (static 2400: 25 %).
So, on this GPU and these programs, the offline-trained DRLCap behaves
like a fixed ~2160 MHz cap plus the case-A rule; online learning then
moves the greedy action around (48.7 % at 2160 MHz, the rest spread over
33 other caps) without making it better.

Hypotheses, NOT tested: (a) with gamma 0.95 the bootstrapped state value
(~70) dwarfs the one-step reward differences between neighbouring caps;
(b) NVML shows a new cap about one sample late (see the sweep section),
so the reward of a_t is partly earned by a_(t-1), which blurs which
action caused it; (c) the Algorithm 1 constants r1..r4, p1..p3, c that
the paper does not publish (ours) favour a middle cap; (d) the Rodinia
training set is small (8 programs) and many states are idle or
host-bound, where every cap gets nearly the same reward.

### Ablations, 5 repeats each

Seed (`results/eval_ablation_seed_overhead_20261031_060047`):

| Policy | Energy saving | Perf. loss | EDP improvement | EDP 95 % CI |
|---|---|---|---|---|
| DRLCap seed 0 | 21.4 % | 5.5 % | 17.1 % | 16.0 – 18.1 |
| DRLCap seed 1 | 20.6 % | 5.7 % | 16.1 % | 14.8 – 17.4 |
| monitor only (seed 0, caps never applied) | 2.7 % | 0.8 % | 1.9 % | 0.8 – 3.1 |

The two seeds agree within noise (they collapsed to 2160 and 2100 MHz).
DRLCap seed 0 is 18.5 % EDP in the 10-repeat run and 17.1 % here: the
difference between two evaluations of the same model is ~1.5 points, so
smaller differences should not be read as real.

Frozen network (`results/eval_ablation_frozen_20261031_082435`, no online
learning, same epsilon 0.1): 27.6 % / 5.4 % / 23.7 % (EDP CI 22.7 – 24.7),
better than the same model with online learning (18.5 % in the main run).
Since the frozen greedy action is always 2160 MHz, this mainly shows that
a fixed ~2160 MHz cap is a good setting on these programs, and that online
learning at test time moved the policy AWAY from it.

### Overhead (paper Sec. VI-C, Fig. 11/12)

Controller per job (main run): 1.0-4.2 s CPU time, median step 8-15 ms of
the 300 ms period (tabular RL 25-47 ms, because more frequent cap changes
each call `nvidia-smi`). The evaluation process's peak resident memory
grows from 640 MB to 1569 MB over the 400 agent jobs of the main run
(~2.3 MB per job; one fresh agent is built per job, so this looks like
TensorFlow/Keras memory not released; NOT investigated, and a deployed
single-agent controller would not rebuild its model per job). CPU energy is not measured (deviation 11).

Monitor-only runs (sampling + agent + learning, caps never applied; the
logged clocks stay at 2500-2700 MHz) used 2.7 % LESS GPU energy than the
plain baseline, at +0.8 % time, consistently on 9 of 10 programs. A
monitoring overhead should go the other way. This also means the agent
policies may get a ~2-3 % head start over the static/COORD policies,
which run without the loop. Diagnostic: see below.

`diag_monitor_effect.py` (`results/diag_monitor_effect_20261031_094448.csv`,
3 repeats, interleaved, rotating order) compares the plain baseline with
(poll) the same job plus a thread that only calls `gpu_control.read()`
every 300 ms, and (monitor) the full monitor-only agent:

| Program | poll / base energy | monitor / base energy | poll / base time | monitor / base time |
|---|---|---|---|---|
| b+tree | 0.975 | 0.984 | 1.000 | 1.004 |
| lud | 0.961 | 0.970 | 0.999 | 0.998 |
| myocyte | 0.962 | 0.985 | 0.999 | 1.006 |
| heartwall | 0.961 | 0.966 | 1.001 | 1.003 |
| vgg16 | 0.961 | 0.966 | 1.003 | 1.010 |

Fact: NVML polling alone gives the 2.5-4 % lower measured energy at the
same runtime; the agent's CPU work adds nothing to it (monitor is if
anything slightly closer to the baseline). Whether the GPU really draws
less power while it is being polled, or the NVML energy counter behaves
differently when polled, cannot be told apart without an external power
meter. Consequence for the comparison above: every agent policy
(DRLCap, variants, RL) ran with 300 ms polling and the baseline, static
caps and COORD did not, so the agent policies' savings include ~3 points
that are not due to capping. A fair re-run would poll NVML at 300 ms in
every policy, baseline included; with that correction DRLCap would be
expected to fall slightly BEHIND the static 2400 MHz cap rather than tie
it (expectation, not measured).

## Results v3: faithful reproduction with fair measurement, 2026-10-31

Changes vs. v2 (commit b3d5206): (1) the reward of action a_t is computed
on samples t+1 -> t+2 (`drlcap.REWARD_DELAY = 1`), because a cap change
shows up mostly one sample late (clock-change slope 0.29 in the next
sample, 0.53 one sample later, from 4609 cap changes under load in the v2
logs); (2) baseline, static and COORD jobs poll NVML every 300 ms like
the agent loop. All five models retrained. Log
`results/queue_v3_20261031_114355.log`, 2026-10-31 11:43 -> 11-01 03:18,
no failed job. Figures: `results/plots_v3_20261101_031739/`.

### Main comparison, 10 repeats (`results/eval_main_v3_20261031_135056`)

| Policy | Energy saving | Perf. loss | EDP improvement | EDP 95 % CI |
|---|---|---|---|---|
| static 2100 MHz | 28.3 % | 8.4 % | 22.3 % | 21.7 – 22.9 |
| static 2400 MHz | 21.3 % | 3.2 % | 18.7 % | 18.0 – 19.4 |
| COORD-like | 27.6 % | 6.6 % | 22.8 % | 22.1 – 23.5 |
| **DRLCap (frequency)** | **17.4 %** | **3.0 %** | **14.9 %** | 14.0 – 15.8 |
| DRLCap power | 2.9 % | 1.4 % | 1.5 % | 0.5 – 2.5 |
| DRLCap hybrid | 13.3 % | 3.2 % | 10.6 % | 9.7 – 11.5 |
| RL (tabular double Q) | 18.3 % | 4.7 % | 14.4 % | 13.7 – 15.2 |

DRLCap per program (energy / perf. loss / EDP, %): b+tree 12.9 / 0.8 /
12.2, dwt2d 9.8 / 4.2 / 6.0, heartwall 23.8 / 2.0 / 22.3, lavaMD 2.2 /
0.8 / 1.4, lud 19.4 / 10.7 / 10.8, myocyte 22.2 / 3.3 / 19.6,
streamcluster 19.0 / -0.8 / 19.4, vgg16 20.9 / 4.0 / 17.8, googlenet
21.2 / 3.0 / 18.8, resnet50 19.8 / 2.1 / 18.2.

Facts:
- With polling in every policy, the static and COORD numbers are
  unchanged from v2 (within 0.5 points), while DRLCap and RL each lose
  ~3.5 points EDP. DRLCap is now significantly BELOW the static 2400 MHz
  cap (CIs do not overlap), and ~8 points EDP below COORD and static
  2100. The paper's claim (DRLCap beats COORD and RL) is not reproduced.
- The policies still collapse (`results/probe_policy_v3_20261101_031735.txt`):
  freq_s0 2415 MHz in 95.8 % of random states, freq_s1 1950 MHz in 100 %,
  hybrid 2475 MHz / 285 W in 99.2 %, power 240 W / 145 W. The reward
  delay did not fix the collapse. In the main run DRLCap's mean cap is
  2300-2400 MHz on every program except lavaMD (2674 MHz: case A forced
  the maximum cap in 78.9 % of its steps, energy saving 2.2 %).
- The power variant does nothing on Rodinia (-0.2 % energy; draw below
  the 100 W minimum limit) and saves 9.6 % on the CNNs.

### Ablations, 5 repeats each

| Run | Energy saving | Perf. loss | EDP improvement | EDP 95 % CI |
|---|---|---|---|---|
| DRLCap seed 0 (online) | 17.3 % | 3.2 % | 14.7 % | 13.3 – 16.1 |
| DRLCap seed 1 (online) | 18.9 % | 6.7 % | 13.6 % | 12.5 – 14.5 |
| monitor only (caps never applied) | -1.0 % | 0.3 % | -1.3 % | -2.4 – -0.2 |
| DRLCap seed 0 frozen (no online learning) | 17.7 % | 2.2 % | 15.9 % | 14.9 – 16.9 |

- Monitor-only now costs 1.0 % energy (EDP -1.3 %) relative to an equally
  polled baseline: this is the controller's own GPU-side overhead
  (paper Fig. 11 analogue), the expected sign, small.
- Seeds agree on EDP within noise but collapsed to different caps (2415 vs
  1950 MHz), visible in the slowdown (3.2 % vs 6.7 %).
- Frozen vs. online: within ~1 point; online learning neither helps nor
  clearly hurts here.

### Diagnosis: why DRLCap does not beat static caps (2026-11-01)

From the v2/v3 training logs (`steps.csv`):
1. **The Algorithm 1 / Eq. 1 reward is level-blind.** dp and df are
   changes between consecutive samples (paper: "during the current and
   the previous sample period"; authors' code: `fre = fre_g_ - fre_g`).
   Holding ANY cap gives df = dp = 0 -> the top score. Mean reward when
   the cap was unchanged 4.01, raised 3.15, lowered 2.33; at a steady
   cap, case C gives 5.8 at 1800-2000 MHz and 5.3 at 2600-2800 MHz, case D
   0.15 at both. The reward therefore penalises changing the cap but does
   not prefer an energy-efficient level, and the learned policy "hold one
   arbitrary cap" is optimal for it.
2. **Case D carries almost no signal.** r = g * u_mem with u_mem as a
   fraction; the median memory utilisation in case D is 1 %, so r ~ 0.05
   (mean |r| 0.26 vs. 4-8 in cases A-C). Case D is the most common case
   at test time (e.g. lud 1064 / 1304 steps, googlenet 2091 / 3577).
3. **Case C is amplified** (g / u_core, mean |r| 8.3, std 7.3, clipped at
   10), so the reward scale differs by >30x between cases.
4. **Case A forces the maximum cap** whenever a sample shows 100 % core /
   0 % memory utilisation (lavaMD 79 % of steps), forfeiting the savings
   the static sweep shows for compute-bound programs.
Causes 1-2 are properties of the reward as published (and of the
authors' code), not of this implementation. See "Results v4" for the
repaired reward (a documented deviation); v3 remains the faithful result.

## Results v4: repaired DRLCap (final, kept), 2026-11-02

v4 keeps everything from v3 (reward delay, polling in every policy) and
changes four things to fix the causes found above. All four are
deviations from the paper (Deviations 15-18 below); v3 is the faithful
result, v4 is the best this implementation achieves.

1. **Reward** `reward.REWARD_MODE = "edp_local"`: idle samples (case B)
   keep the paper's c / cap; otherwise
   r = -3 log((P + Pbar) / 285 W / rate), clipped to +-10, with
   rate = u_core * f / 2745 MHz if u_core >= 50 % and rate = 1 if the GPU
   is host-bound, Pbar = the job's running mean power. Minimising E * T
   over phases means minimising (P + Pbar) per unit of work, so this is a
   local form of the EDP objective. Checked offline against a per-sample
   sweep of the 8 training programs (`results/reward_sweep_20261101_085612*`,
   `reward_design.py`): the cap it ranks best is on average 2.0 % EDP
   worse than each program's EDP-best cap (max 14.2 %, Spearman 0.76),
   compared with 9.5 % (Spearman 0.24) for Algorithm 1 / Eq. 1.
2. **No forced maximum in case A** (`drlcap.FORCE_CASE_A = False`): the
   sweep shows the maximum cap is the worst EDP choice for compute-bound
   programs on the 4070 Ti.
3. **gamma 0.2** instead of 0.95 (Table V): a cap only affects the next
   one or two samples. Replaying the logged v3 transitions with gamma 0.95
   gave one constant high cap; with gamma 0 - 0.2 the greedy cap depended
   on the state (`offline_gamma.py`).
4. **Fewer, wider actions**: 8 frequency caps 1800-2745 MHz (step 135),
   8 power caps 100-285 W, 4 x 4 hybrid pairs, instead of 64 caps 15 MHz
   apart; with 5000 training steps each of 64 caps is tried only a few
   dozen times. The tabular RL baseline uses the same reward, gamma and
   8 caps, so the comparison stays like-for-like.

Log `results/queue_v4_20261101_110400.log`, 2026-11-01 11:04 -> 11-02
02:45, no failed job. Figures: `results/plots_v4_20261102_024521/`.
Training runs `results/train_{freq_s0,freq_s1,power_s0,hybrid_s0,rl_s0}_v4_*`.

### Main comparison, 10 repeats (`results/eval_main_v4_20261101_131118`)

| Policy | Energy saving | Perf. loss | EDP improvement | EDP 95 % CI | v3 EDP |
|---|---|---|---|---|---|
| static 2100 MHz | 28.3 % | 8.2 % | 22.4 % | 21.7 – 23.1 | 22.3 % |
| static 2400 MHz | 20.9 % | 3.8 % | 17.9 % | 17.2 – 18.6 | 18.7 % |
| COORD-like | 27.5 % | 6.9 % | 22.5 % | 21.8 – 23.2 | 22.8 % |
| **DRLCap (frequency)** | **19.8 %** | **5.2 %** | **15.6 %** | 14.5 – 16.8 | 14.9 % |
| DRLCap power | 0.2 % | 0.8 % | -0.6 % | -1.7 – 0.4 | 1.5 % |
| DRLCap hybrid | 21.8 % | 12.7 % | 11.9 % | 10.8 – 13.1 | 10.6 % |
| RL (tabular double Q) | 19.1 % | 6.7 % | 13.7 % | 13.0 – 14.4 | 14.4 % |

Rodinia / CNN split (energy / slowdown / EDP, %): DRLCap 19.7 / 6.0 / 14.8
and 20.1 / 3.2 / 17.5; COORD 25.6 / 6.0 / 21.1 and 31.6 / 8.9 / 25.5;
static 2100 27.1 / 9.6 / 20.0 and 31.1 / 4.9 / 27.7.

DRLCap per program (energy / perf. loss / EDP, %; mean cap): b+tree
20.1 / 2.3 / 18.3 (1953 MHz), dwt2d 17.0 / 10.7 / 8.1 (1859), heartwall
23.8 / 2.2 / 22.2 (2277), lavaMD 19.9 / 9.7 / 12.2 (2424), lud 19.7 /
10.0 / 11.6 (2340), myocyte 17.3 / 5.0 / 13.2 (2418), streamcluster
19.6 / 2.6 / 17.2 (2244), vgg16 20.2 / 4.1 / 17.0 (2213), googlenet
21.1 / 4.3 / 17.8 (2207), resnet50 18.8 / 1.3 / 17.8 (2281).

Facts:
- DRLCap is still significantly below COORD and static 2100 MHz (about
  7 points EDP, CIs do not overlap) and below static 2400 MHz. It beats
  static 2100 on one program (lavaMD 12.2 vs 11.0) and COORD on none.
  The paper's claim (DRLCap beats COORD and RL) is not reproduced.
- Versus v3: +0.7 points EDP (CIs overlap, so not a clear gain); energy
  saving +2.4 points, slowdown +2.2 points. lavaMD improves most
  (EDP 1.4 -> 12.2) because case A no longer forces the maximum cap;
  myocyte gets worse (19.6 -> 13.2).
- DRLCap now beats the RL baseline (15.6 vs 13.7, CIs do not overlap);
  in v3 they were equal.
- **The policy adapts, but its level is too high when the GPU is busy.**
  Caps chosen during the main run (`steps.csv`): idle samples mean 2102
  MHz (1800 in 45 %), host-bound (u_core < 50 %) 2130 MHz (1800 in 43 %),
  GPU-busy (u_core >= 50 %) 2379 MHz (2475 30 %, 2610 24 %, 2340 23 %).
  The static sweep and static 2100 suggest ~2100 MHz for the busy phases
  of these programs. The RL baseline does not adapt (mean 2266-2270 MHz
  in all three classes, Fig. 5 almost uniform over the 8 caps).
- Policy probe (`results/probe_policy_v4_*.txt`): freq_s0 5 distinct
  greedy actions (2475 43 %, 1800 36 %, 2610 14 %), freq_s1 5 (2340 45 %,
  2205 30 %, 1935 20 %); v3 collapsed to one cap in 96-100 % of random
  states. Median best-vs-second Q gap is small (0.08 / 0.11), i.e. the
  network separates neighbouring caps weakly. freq_s0 picks 2475 MHz for
  a synthetic idle state (the validation run picked 1800).
- Training (Fig. 6): the mean chosen action keeps swinging between 0.1
  and 0.8 of the cap range until step 5000; the policy has not settled.
- The power variant does nothing (0.2 % energy): Rodinia draws less than
  the 100 W minimum limit, and it picks 259 W, which does not bind, for
  39 % of steps. The hybrid saves energy but costs 12.7 % time (lud -14.8 %
  EDP at 1800 MHz / 100 W).

### Ablations, 5 repeats each

| Run | Energy saving | Perf. loss | EDP improvement | EDP 95 % CI |
|---|---|---|---|---|
| DRLCap seed 0 (online) | 20.0 % | 5.1 % | 16.0 % | 14.7 – 17.2 |
| DRLCap seed 1 (online) | 23.5 % | 6.5 % | 18.6 % | 17.2 – 20.0 |
| monitor only (caps never applied) | -0.8 % | 0.4 % | -1.2 % | -2.4 – -0.1 |
| DRLCap seed 0 frozen (no online learning) | 19.8 % | 4.1 % | 16.4 % | 15.5 – 17.3 |

- Seed matters: seed 1 is 2.6 points better (it picks 2205 MHz in
  43 % of steps) but still ~4 points below COORD / static 2100. The main
  table uses seed 0 as planned before the run.
- Overhead: monitoring alone costs 0.8 % energy, as in v3 (1.0 %).
- Frozen vs. online: within 0.5 points, as in v3; online learning during
  a 20-100 s test job does not change the result.

### Conclusion of the reproduction

On the RTX 4070 Ti, with the Rodinia + CNN workloads available here,
DRLCap does not reach the static 2100 MHz cap or the COORD-like policy,
neither as published (v3, 14.9 % EDP) nor with the repaired reward, gamma
and action set (v4, 15.6 %; 18.6 % with the second seed). The repairs
made the policy state-dependent (v3 was constant) and lifted it above
the RL baseline, but it still selects caps that are too high in
GPU-busy phases. Hypotheses, not tested: 5000 training steps are too
few for a stable policy (Fig. 6), and the u_core * f work proxy rewards
higher clocks on memory-heavy phases where the clock does not raise
throughput.

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
13. **Reward delay** (v3, v4): the reward of action a_t is computed on
    samples t+1 -> t+2 (`drlcap.REWARD_DELAY = 1`) because a cap change
    shows up mostly one sample late on this GPU.
14. **NVML polling in every policy** (v3, v4): baseline, static and COORD
    jobs poll NVML every 300 ms like the agent, because polling alone
    lowers measured energy by 2.5-4 % (`diag_monitor_effect.py`).

v4 only (see "Results v4"; v3 is the faithful result):

15. **Reward**: EDP-local reward (`reward.REWARD_MODE = "edp_local"`)
    instead of Algorithm 1 / Eq. 1, case B unchanged.
16. **Case A** no longer forces the maximum cap (`drlcap.FORCE_CASE_A`).
17. **gamma 0.2** instead of 0.95 (DRLCap and the RL baseline).
18. **Action sets**: 8 frequency caps (1800-2745 MHz, step 135), 8 power
    caps (100-285 W), 4 x 4 hybrid pairs; the RL baseline uses the 8
    frequency caps.
