# Why v4 differs from the paper (review items 1-2)

v4 (`drlcap_repro`, commit e5a5c26) changed five things relative to the paper.
All results, code and models of v1-v4 are kept unchanged in `drlcap_repro/`.
Each change was made on top of the previous one in a single run, so their
individual effects were **not** measured separately. The reviewer is right
that this must be done, and the new reference configuration (`REFERENCE_CONFIG.md`)
undoes all five, so each can be re-tested on its own.

## The starting point: v3, the faithful run

v3 used the paper's reward (Algorithm 1 + Eq. 1), γ 0.95, Case-A forcing and
64 caps 15 MHz apart. Its result on the 10 test programs (10 repeats) was:
DRLCap 17.4 % energy saving, 3.0 % slowdown, 14.9 % EDP improvement, below
static 2400 MHz (18.7 % EDP) and static 2100 MHz / COORD-like (22.3-22.8 %).

Evidence collected from the v2/v3 training logs (`steps.csv`) and a policy
probe (`probe_policy.py`):

| Observation | Number |
|---|---|
| Trained network picks the same cap in almost every state | freq seed 0: 2415 MHz in 95.8 % of random states; seed 1: 1950 MHz in 100 % |
| Reward when the cap was unchanged / raised / lowered | mean 4.01 / 3.15 / 2.33 |
| Case C reward at a steady cap, low vs high cap | 5.8 (1800-2000 MHz) vs 5.3 (2600-2800 MHz) |
| Case D reward magnitude | mean abs 0.26 (median memory utilisation 1 %) vs 4-8 in cases A-C |
| Case A forcing on lavaMD | 79 % of steps at the maximum cap, 2.2 % energy saving |
| Offline reward check on the 8 training programs | the cap Algorithm 1 ranks best is on average 9.5 % EDP worse than the measured best cap (Spearman 0.24) |

**Our reading (a hypothesis, not proven):** Δp and Δf are changes between two
consecutive samples, so holding *any* cap gets the top score. The reward
punishes changing the cap but does not say which cap level is efficient, so
"hold one cap" is optimal for it, which matches the collapsed policies.

## The five v4 changes and the reason for each

| # | Change | Reason given at the time | Evidence | Was the effect measured alone? |
|---|---|---|---|---|
| 1 | Reward → `edp_local`: r = -3 ln((P + P̄)/285 W / rate), rate = u·f/2745 if u ≥ 50 % else 1; case B kept | Algorithm 1 is level-blind (above) | Offline check: best-ranked cap 2.0 % EDP worse than measured best (Spearman 0.76) vs 9.5 % for Algorithm 1 | Offline only, not in a training run |
| 2 | Case-A forcing off | Max cap is the worst-EDP cap for compute-bound programs on the 4070 Ti | Training-set frequency sweep (EDP-best 2200-2400 MHz for gaussian, hotspot, particlefilter) | No |
| 3 | γ 0.95 → 0.2 | A cap affects only the next 1-2 samples | Offline replay of logged v3 transitions: γ 0.95 → one constant cap; γ 0-0.2 → state-dependent caps (`offline_gamma.py`) | Offline replay only |
| 4 | 64 caps → 8 caps (135 MHz apart) | 5000 steps give each of 64 caps only a few dozen tries | None beyond the count | No |
| 5 | Reward delay 1 sample (introduced in v3, kept) | NVML shows a cap change mostly one sample late | Over 4609 cap changes under load: clock moved 0.29x of the change in the next sample, 0.53x one sample later | No (v2 → v3 also changed NVML polling of the baselines at the same time) |

## What v4 achieved and what is still wrong with it

v4: 19.8 % energy, 5.2 % slowdown, 15.6 % EDP (CI 14.5-16.8), still below
static 2400 MHz (17.9 %), static 2100 MHz (22.4 %) and COORD-like (22.5 %).

A later offline check (`diag_rate_proxy.py`, commit 5aa96cb) found a flaw in
change 1: v4 measures progress as utilisation × clock, but for the CNNs and
heartwall a higher clock speeds the busy phases up by only 0.26-0.52 of the
clock increase (memory-limited), while utilisation stays the same. The reward
therefore prefers 2400-2700 MHz where the measured EDP-best cap is 2100 MHz.
So the reviewer's concern that the rate estimate "may be misleading for
memory-limited or host-limited workloads" is confirmed by our own data.

## Consequence

v4 is a set of untested-in-isolation changes and does not reproduce the paper.
The new folder trains a fresh model under the paper's reference configuration
first, and each v4 change is then a separate one-setting variant with its own
recorded effect (`REFERENCE_CONFIG.md` section 4).
