from simulated_env import SimulatedGPUEnv


env = SimulatedGPUEnv(
    seed=42
)

state = env.reset()

print("\nInitial state:")
print(state)

print("\nState dimension:")
print(len(state))

print("\nActions:")
print(env.n_actions)

for step in range(20):

    action = step % env.n_actions

    next_state, reward, done = env.step(
        action
    )

    metrics = env.get_metrics()

    print(
        f"Step {step + 1:02d} | "
        f"Action {action} | "
        f"Reward {reward:.3f} | "
        f"Power {metrics['power']:.2f} W | "
        f"Freq {metrics['frequency']:.1f} MHz | "
        f"GPU {metrics['gpu_util']:.1f}% | "
        f"Mem {metrics['mem_util']:.1f}% | "
        f"Temp {metrics['temperature']:.2f} C"
    )

    state = next_state

    if done:

        break
