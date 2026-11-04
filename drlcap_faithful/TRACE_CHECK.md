# Transition-timing check (review item 3)

Command: `python drlcap.py trace gaussian --out results/trace_gaussian_delay0.csv`
(reference, reward delay 0) and the same with `--reward-delay 1` (the v3/v4
setting). One `gaussian` job (3 runs, ~10 s), fresh untrained network,
ε = 0.3, no learning. Every row has a millisecond timestamp from job start.
Events: `sample` (NVML read), `act` (ε-greedy choice), `apply_start` /
`apply_done` (the `nvidia-smi -lgc` call), `reward`, `replay_insert`,
`dropped`.

## Reference configuration (delay 0): correct

```
    13.0 sample        sample 0  util 0  mem 0  P 9.4 W  clock 210
    18.5 act           decision 0 -> cap 2220 (greedy)
    66.1 apply_done    cap 2220 in place (nvidia-smi took 48 ms)
   310.2 sample        sample 1  util 3  mem 0
   310.4 reward        decision 0, sample pair 0->1, r = 10.0 (case C, clipped)
   310.4 replay_insert (s0, a0, r, s1), is_end 0
   ...
 10821.0 sample        sample 36, job finished
 10821.1 reward        decision 35, pair 35->36, r = 5.05 (case C)
 10821.2 replay_insert (s35, a35, r, s36), is_end 1
```

* Decision t is made right after sample t; its cap is in place 40-50 ms later.
* Its reward uses samples t -> t+1 (Algorithm 1 lines 2-4) and it is stored
  as (s_t, a_t, r, s_t+1). State, action, reward and next state line up.
* The last decision of the job is stored with `is_end = 1`; nothing is dropped.
* Samples are 300 ms apart (310, 611, 911 ... ms).

## v3/v4 setting (delay 1): two defects confirmed

```
   607.1 reward        decision 0, pair 1->2
   607.2 replay_insert (s0, a0, r, s1)          <- reward is from 1->2, next state is s1
 9915.9 reward        decision 31, pair 32->33
 9916.0 replay_insert (s31, a31, r, s32), is_end 1
 9916.0 dropped       decision 32               <- last decision never stored
```

1. **Reward and next state are one sample apart.** The reward of a_t comes
   from samples t+1 -> t+2, but the stored next state is s_t+1. The Q-target
   r + γ max Q(s_t+1, ·) therefore mixes a reward from one interval with a
   bootstrap from an earlier state.
2. **End of job.** The final decision is dropped, and the `is_end` flag is put
   on the second-to-last decision instead of the last one.

Both are reasons the reference configuration uses delay 0. Delay 1 stays a
separate variant only, and if it is ever used, these two defects must be
fixed first.

## Other facts visible in the trace

* **Case C amplification.** At sample 1 the GPU shows 3 % utilisation and 0 %
  memory utilisation, so case C gives r = g / 0.03 = 167, which is clipped to 10.
  This is open question Q1/Q2 in `REFERENCE_CONFIG.md`.
* **NVML lag at the end.** The sample taken after the job has ended still
  shows 99 % utilisation (NVML averages over its last window), so the
  terminal transition's next state looks busy.
* **Power reading lag.** Board power stays at 9.358 W for the first few
  samples while utilisation changes, so Δp is 0 there.
