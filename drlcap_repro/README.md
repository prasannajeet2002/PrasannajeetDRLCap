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
| `drlcap.py` | 300 ms runtime control loop; `python drlcap.py train` = offline training |
| `evaluate.py` | Comparison with the NVIDIA default policy: energy, runtime, EDP, geomean (Sec. V-D) |

Results go to `results/` with timestamped names; nothing is overwritten.

## Commands

```bash
# 1. static cap sweep (motivation + action range)
python frequency_sweep.py --workloads hotspot gaussian nw bfs particlefilter

# 2. offline training on the training set (5000 iterations, Table V)
python drlcap.py train

# 3. test on unseen workloads vs. NVIDIA default, online learning on
python evaluate.py --model results/train_drlcap_<stamp>/model_final.keras --repeats 3
```

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
3. **No CNN test workloads**: TensorFlow 2.21 in `prasannajeet_env` cannot
   create a cuDNN handle (system cuDNN 8.9 vs. required 9.x for CUDA 12;
   the pip cuDNN targets CUDA 13), and the installed PyTorch needs a newer
   driver. VGG-16 / GoogleNet tests are therefore missing.
4. **Unpublished constants**: r1..r4, p1..p3, g1..g4, n, c, the hidden
   layer width, PER alpha/beta are not given in the paper; our values are
   in `reward.py` / `agent.py` and saved in each run's `config.json`.
5. **Reward clipping** to +-5 (case C divides by utilisation).
6. **Case A** is implemented by forcing the next action to the maximum cap
   when the latest sample is fully compute-bound (u_core = 100,
   u_mem = 0); the reward for that step then bins the power change.
7. **Energy** is the NVML board energy counter (joules), not
   power x time; the paper does not state its energy source.
8. **Repeats**: the paper runs each test 10 times; `--repeats` defaults to 3
   to fit the available time.
