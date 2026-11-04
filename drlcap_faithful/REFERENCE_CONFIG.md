# DRLCap paper-aligned reference configuration (proposal for review)

This folder (`drlcap_faithful/`) is a fresh copy of the controller code from
`drlcap_repro/`, set to the paper's method. Nothing in `drlcap_repro/` or its
results has been changed. Every setting below is marked with its source:

* **Paper**: stated in Wang et al., IEEE TSUSC 9(5), 2024 (section/table given).
* **Authors' code**: the TF2 port of the authors' V100S scripts in the project
  root (`Freq_DDQN_prioritized_save_tf2.py`), used only where the paper is silent.
* **Our choice**: neither source gives it; the value is a proposal to review.

## 1. Reward (paper Algorithm 1 + Eq. 1), `reward.py`, `REWARD_MODE = "alg1"`

| Item | Value | Source |
|---|---|---|
| Cases | A: u_mem = 0 and u_core = 100; B: u_mem = 0 and u_core = 0; C: u_mem = 0 otherwise; D: u_mem > 0 | Paper Alg. 1 |
| Case A | force the maximum cap (`FORCE_CASE_A = True`), reward from power change Δp binned at p1 < p2 < p3 into r1 > r2 > r3 > r4 | Paper Alg. 1 lines 6-13 |
| p1, p2, p3 | 5, 15, 30 W | **Our choice** (paper gives no values) |
| r1..r4 | 5, 2.5, 0, -2.5 | **Our choice** (r1 = g1) |
| Case B | r = c / action, c = 5 x lowest cap, so r is in (0, 5] | Paper Alg. 1 line 16; **c is our choice** |
| Case C | r = g / u_core | Paper Alg. 1 line 20 |
| Case D | r = g x u_mem | Paper Alg. 1 line 25 |
| g (Δp <= 0) | +5 / -1 / -2 / -3 | Authors' code |
| g (Δp > 0) | -1 / -2 / -3 / -4 | Authors' code |
| Eq. 1 bin width n x step | 45 MHz (n = 3, step = 15 MHz = 4070 Ti hardware step) | Paper Eq. 1 form; 45 MHz from authors' code |
| Power before differencing | floored to whole watts | Authors' code |
| Utilisation units in C and D | fractions in [0, 1] (NVML % / 100) | **Our choice** (paper does not say; see open question Q1) |
| Clipping | reward clipped to [-10, +10] | **Our choice** (paper silent; see Q2) |
| Reward timing | reward of action a_t from samples t -> t+1 (`REWARD_DELAY = 0`) | Paper Alg. 1 lines 2-4 |

## 2. Agent (paper Table V and Sec. IV-D), `agent.py`

| Item | Value | Source |
|---|---|---|
| Algorithm | Double DQN + prioritized replay (sum-tree) | Paper Alg. 2 |
| Batch size | 32 | Paper Table V |
| Learning rate | 0.001 | Paper Table V |
| Replay buffer | 512 | Paper Table V |
| Discount γ | **0.95** | Paper Table V |
| ε | 0.3 -> 0.1 (linear over training), fixed in online deployment | Paper Table V, Sec. IV-D |
| Training iterations | 5000 | Paper Table V |
| Target-network update | every 8 updates | Paper Table V |
| Network | 5 inputs, 1 hidden layer, one output per action | Paper Fig. 8 |
| Hidden units / activation | 20, ReLU | Authors' code |
| Init / optimizer | N(0, 0.3) weights, bias 0.1, RMSprop | Authors' code |
| PER α, β, ε, priority cap | 0.6; 0.4 rising 0.001 per batch; 0.01; 1.0 | Authors' code (α, β form also in paper) |
| Learning starts | offline: when the buffer is full; online: after one batch | Authors' code / our choice for online |
| Terminal flag | y = r at the end of a job (`is_end`) | Paper Alg. 2 line 12 |

## 3. State, actions, sampling, `drlcap.py`, `gpu_control.py`

| Item | Value | Source |
|---|---|---|
| State | GPU util, memory util, power, SM clock, temperature (NVML) | Paper Sec. IV-B |
| Discretisation | util 20 buckets, mem 20, power 20 (10-285 W), temp 30 (30-90 C), clock as index on the 15 MHz ladder; scaled to [0, 1) | Bucket layout from authors' code; ranges and scaling **our choice** |
| Sampling interval | 300 ms | Paper Sec. VI-C |
| Knob | frequency cap via `nvidia-smi -lgc 210,<cap>` on GPU 1 | Paper Sec. V-A (nvidia-smi / pynvml) |
| Actions (reference) | 1800-2745 MHz in 15 MHz hardware steps (64 caps) | Paper Table IV uses the hardware step; the range is **our choice**, based on the 4070 Ti frequency sweep (V100S range ratio 1035/1597 = 0.65, ours 1800/2745 = 0.66) |
| Cap applied | only when the chosen cap changes | Paper Sec. VI-C |
| Agent placement | CPU process, TensorFlow on CPU | Paper Sec. V-A |

## 4. Variants kept separate (one setting changed per run)

Each can be switched on alone with a command-line flag, so its effect is
recorded on its own (review item 2):

| Variant | Flag | What drlcap_repro used |
|---|---|---|
| EDP-local reward | `--reward-mode edp_local` | v4 |
| γ = 0.2 | `--gamma 0.2` | v4 |
| no Case-A forcing | `--no-force-case-a` | v4 |
| one-sample reward delay | `--reward-delay 1` | v3, v4 |
| 8 coarse caps (135 MHz apart) | `--variant freq8` | v4 |

## 5. Open questions for the reviewer

* **Q1, utilisation units.** Case C divides by u_core and case D multiplies by
  u_mem. With fractions (our proposal) case C ranges from g (u = 100 %) to
  100 g (u = 1 %); with percentages it ranges from g/100 to g. The paper does
  not say which. The choice changes how much case C and D samples weigh.
* **Q2, clipping.** Without clipping, case C at 1 % utilisation gives rewards
  up to 500, which dominate the squared TD loss. We propose ±10 and will
  record how often clipping triggers.
* **Q3, constants not in the paper**: p1-p3, r1-r4 and c (table above).
* **Q4, action range** 1800-2745 MHz (64 caps), versus the coarse 8-cap set,
  to be compared with the same training budget (review item 4).
