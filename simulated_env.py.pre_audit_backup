import numpy as np


class SimulatedGPUEnv:

    """

    State:
        0 - GPU utilization
        1 - Memory utilization
        2 - GPU power
        3 - GPU frequency
        4 - Temperature
        5 - Workload intensity
        6 - Power cap

    Action:
        Select one of 8 GPU power-cap configurations.
    """

    def __init__(self, seed=None):

        # ============================================================
        # RANDOM NUMBER GENERATOR
        # ============================================================

        self.rng = np.random.default_rng(seed)

        # ============================================================
        # GPU PARAMETERS
        # ============================================================

        self.min_power = 100.0
        self.max_power = 250.0

        self.min_clock = 1000.0
        self.max_clock = 2500.0

        self.ambient_temperature = 30.0

        self.target_temperature = 75.0
        self.warning_temperature = 83.0
        self.critical_temperature = 90.0

        # ============================================================
        # DISCRETE POWER ACTIONS
        # ============================================================

        # 8 discrete actions.
        # They approximate different GPU power-cap configurations.

        self.power_caps = np.array([
            120.0,
            140.0,
            160.0,
            180.0,
            200.0,
            220.0,
            235.0,
            250.0
        ], dtype=np.float32)

        self.actions = list(range(len(self.power_caps)))

        self.n_actions = len(self.actions)

        # ============================================================
        # WORKLOAD TYPES
        # ============================================================

        self.workload_types = [
            "idle",
            "light",
            "compute",
            "memory",
            "mixed"
        ]

        # ============================================================
        # EPISODE SETTINGS
        # ============================================================

        self.max_steps = 100

        self.step_count = 0

        # ============================================================
        # STATE VARIABLES
        # ============================================================

        self.gpu_util = 0.0
        self.mem_util = 0.0
        self.power = 0.0
        self.frequency = 0.0
        self.temperature = 0.0

        self.workload = 0.0
        self.workload_type = "light"

        self.power_cap = 200.0

        self.current_action = 4
        self.previous_action = 4

        # ============================================================
        # PREVIOUS VALUES
        # ============================================================

        self.previous_power = 0.0
        self.previous_frequency = 0.0

        # ============================================================
        # LOGGING VARIABLES
        # ============================================================

        self.last_reward = 0.0
        self.last_delta_power = 0.0
        self.last_delta_frequency = 0.0

        # ============================================================
        # RESET
        # ============================================================

        self.reset()

    # =================================================================
    # RESET
    # =================================================================

    def reset(self):

        self.step_count = 0

        # Start near a reasonable middle configuration.

        self.power_cap = 200.0

        self.current_action = 4
        self.previous_action = 4

        # Random workload phase.

        self.workload_type = self.rng.choice(
            self.workload_types
        )

        self.workload = self._initial_workload()

        # Initial frequency.

        self.frequency = 2000.0

        # Initial temperature.

        self.temperature = 45.0

        # Initial utilization.

        self._update_utilization()

        # Initial power.

        self._update_power()

        # Store previous values.

        self.previous_power = self.power
        self.previous_frequency = self.frequency

        return self._get_state()

    # =================================================================
    # INITIAL WORKLOAD
    # =================================================================

    def _initial_workload(self):

        if self.workload_type == "idle":
            return self.rng.uniform(0.00, 0.08)

        if self.workload_type == "light":
            return self.rng.uniform(0.15, 0.40)

        if self.workload_type == "compute":
            return self.rng.uniform(0.70, 1.00)

        if self.workload_type == "memory":
            return self.rng.uniform(0.65, 0.95)

        if self.workload_type == "mixed":
            return self.rng.uniform(0.50, 0.90)

        return 0.5

    # =================================================================
    # DYNAMIC WORKLOAD
    # =================================================================

    def _update_workload(self):

        # Workload phases do NOT change every timestep.
        # This is important for realistic temporal behavior.

        phase_change_probability = 0.06

        if self.rng.random() < phase_change_probability:

            self.workload_type = self.rng.choice(
                self.workload_types
            )

            target = self._initial_workload()

        else:

            # Gradually vary current workload.

            target = self.workload + self.rng.normal(
                0.0,
                0.05
            )

        # Smooth transition.

        self.workload += (
            0.15 *
            (target - self.workload)
        )

        self.workload = np.clip(
            self.workload,
            0.0,
            1.0
        )

    # =================================================================
    # UTILIZATION MODEL
    # =================================================================

    def _update_utilization(self):

        frequency_ratio = (
            self.frequency /
            self.max_clock
        )

        power_ratio = (
            self.power_cap /
            self.max_power
        )

        # -------------------------------------------------------------
        # IDLE
        # -------------------------------------------------------------

        if self.workload_type == "idle":

            target_gpu = (
                2.0 +
                10.0 * self.workload
            )

            target_mem = (
                2.0 +
                8.0 * self.workload
            )

        # -------------------------------------------------------------
        # LIGHT
        # -------------------------------------------------------------

        elif self.workload_type == "light":

            target_gpu = (
                15.0 +
                45.0 * self.workload +
                8.0 * frequency_ratio
            )

            target_mem = (
                10.0 +
                35.0 * self.workload
            )

        # -------------------------------------------------------------
        # COMPUTE
        # -------------------------------------------------------------

        elif self.workload_type == "compute":

            target_gpu = (
                60.0 +
                38.0 * self.workload +
                5.0 * frequency_ratio
            )

            target_mem = (
                15.0 +
                30.0 * self.workload
            )

        # -------------------------------------------------------------
        # MEMORY
        # -------------------------------------------------------------

        elif self.workload_type == "memory":

            target_gpu = (
                30.0 +
                40.0 * self.workload
            )

            target_mem = (
                55.0 +
                42.0 * self.workload
            )

        # -------------------------------------------------------------
        # MIXED
        # -------------------------------------------------------------

        else:

            target_gpu = (
                40.0 +
                45.0 * self.workload +
                5.0 * frequency_ratio
            )

            target_mem = (
                40.0 +
                45.0 * self.workload
            )

        # -------------------------------------------------------------
        # POWER LIMIT EFFECT
        # -------------------------------------------------------------

        # Lower power cap should gradually restrict achievable
        # compute throughput.

        power_effect = (
            0.75 +
            0.25 * power_ratio
        )

        target_gpu *= power_effect

        # -------------------------------------------------------------
        # THERMAL THROTTLING
        # -------------------------------------------------------------

        if self.temperature > self.warning_temperature:

            thermal_excess = (
                self.temperature -
                self.warning_temperature
            )

            thermal_factor = max(
                0.65,
                1.0 -
                0.025 * thermal_excess
            )

            target_gpu *= thermal_factor

        # -------------------------------------------------------------
        # SMOOTH TRANSITION
        # -------------------------------------------------------------

        self.gpu_util += (
            0.30 *
            (target_gpu - self.gpu_util)
        )

        self.mem_util += (
            0.30 *
            (target_mem - self.mem_util)
        )

        # Small measurement noise.

        self.gpu_util += self.rng.normal(
            0.0,
            0.8
        )

        self.mem_util += self.rng.normal(
            0.0,
            0.8
        )

        self.gpu_util = np.clip(
            self.gpu_util,
            0.0,
            100.0
        )

        self.mem_util = np.clip(
            self.mem_util,
            0.0,
            100.0
        )

    # =================================================================
    # FREQUENCY MODEL
    # =================================================================

    def _update_frequency(self):

        power_ratio = (
            self.power_cap /
            self.max_power
        )

        workload_factor = (
            0.90 +
            0.10 * self.workload
        )

        # Base frequency allowed by power cap.

        target_frequency = (
            self.min_clock +
            (
                self.max_clock -
                self.min_clock
            )
            * power_ratio
        )

        target_frequency *= workload_factor

        # -------------------------------------------------------------
        # THERMAL CONTROL
        # -------------------------------------------------------------

        if self.temperature > self.warning_temperature:

            thermal_excess = (
                self.temperature -
                self.warning_temperature
            )

            thermal_factor = max(
                0.70,
                1.0 -
                0.015 * thermal_excess
            )

            target_frequency *= thermal_factor

        # Smooth frequency transition.

        self.frequency += (
            0.25 *
            (target_frequency - self.frequency)
        )

        # Small clock variation.

        self.frequency += self.rng.normal(
            0.0,
            8.0
        )

        self.frequency = np.clip(
            self.frequency,
            self.min_clock,
            self.max_clock
        )

    # =================================================================
    # POWER MODEL
    # =================================================================

    def _update_power(self):

        utilization = (
            self.gpu_util /
            100.0
        )

        memory = (
            self.mem_util /
            100.0
        )

        frequency_ratio = (
            self.frequency /
            self.max_clock
        )

        # -------------------------------------------------------------
        # STATIC POWER
        # -------------------------------------------------------------

        static_power = 35.0

        # -------------------------------------------------------------
        # COMPUTE POWER
        # -------------------------------------------------------------

        # Nonlinear frequency relationship.
        # This is a simulation model, not a claim of an exact
        # hardware power equation.

        compute_power = (
            85.0 *
            utilization *
            (
                0.40 +
                frequency_ratio
            ) ** 2
        )

        # -------------------------------------------------------------
        # MEMORY POWER
        # -------------------------------------------------------------

        memory_power = (
            25.0 *
            memory
        )

        # -------------------------------------------------------------
        # TOTAL
        # -------------------------------------------------------------

        calculated_power = (
            static_power +
            compute_power +
            memory_power
        )

        # Hardware power cannot exceed the configured cap.

        self.power = np.clip(
            calculated_power +
            self.rng.normal(0.0, 1.5),
            35.0,
            self.power_cap
        )

    # =================================================================
    # THERMAL MODEL
    # =================================================================

    def _update_temperature(self):

        # Heat generated by power.

        heat_generation = (
            0.020 *
            self.power
        )

        # Cooling increases with difference from ambient.

        cooling = (
            0.028 *
            (
                self.temperature -
                self.ambient_temperature
            )
        )

        thermal_delta = (
            heat_generation -
            cooling
        )

        # Thermal inertia.

        self.temperature += (
            0.25 *
            thermal_delta
        )

        # Small thermal noise.

        self.temperature += self.rng.normal(
            0.0,
            0.05
        )

        self.temperature = np.clip(
            self.temperature,
            self.ambient_temperature,
            100.0
        )

    # =================================================================
    # DRLCAP-INSPIRED SCORE g
    # =================================================================

    def _calculate_g(self, delta_power, delta_frequency):

        # Frequency-step scale used by our simulator.
        #
        # DRLCap uses a hardware-dependent step in its
        # piecewise reward calculation. Our simulator uses
        # 75 MHz as an approximate control-resolution scale.

        freq_step = 75.0

        # ============================================================
        # CASE 1: POWER DECREASE / STABILITY
        # ============================================================

        if delta_power <= 0.0:

            # Small frequency loss while reducing power:
            # very desirable.

            if delta_frequency >= -freq_step:

                return 1.00

            elif delta_frequency >= -2.0 * freq_step:

                return 0.75

            elif delta_frequency >= -3.0 * freq_step:

                return 0.45

            else:

                return 0.15

        # ============================================================
        # CASE 2: POWER INCREASE
        # ============================================================

        else:

            # Increasing power is normally undesirable.
            # If frequency also increases substantially,
            # the penalty is smaller because the GPU is obtaining
            # additional performance.

            power_ratio = min(
                delta_power / 20.0,
                1.0
            )

            frequency_ratio = np.clip(
                delta_frequency / 150.0,
                -1.0,
                1.0
            )

            score = (
                0.20 *
                frequency_ratio
                -
                0.70 *
                power_ratio
            )

            return float(
                np.clip(
                    score,
                    -1.0,
                    0.20
                )
            )

    # =================================================================
    # DRLCAP-INSPIRED REWARD
    # =================================================================

    def _calculate_reward(self):

        delta_power = (
            self.power -
            self.previous_power
        )

        delta_frequency = (
            self.frequency -
            self.previous_frequency
        )

        # -------------------------------------------------------------
        # CASE CLASSIFICATION
        # -------------------------------------------------------------

        is_idle = (
            self.gpu_util < 8.0
            and
            self.workload < 0.12
        )

        is_compute_bound = (
            self.gpu_util > 80.0
            and
            self.mem_util < 35.0
        )

        is_memory_bound = (
            self.mem_util > 70.0
        )

        # -------------------------------------------------------------
        # CASE A: HIGHLY COMPUTE INTENSIVE
        # -------------------------------------------------------------

        if is_compute_bound:

            # Reward stable/decreasing power.
            # Strongly penalize unnecessary power increase.

            if delta_power <= 0:

                if delta_power >= -3:

                    reward = 1.00

                elif delta_power >= -8:

                    reward = 0.85

                elif delta_power >= -15:

                    reward = 0.65

                else:

                    reward = 0.40

            else:

                if delta_power <= 3:

                    reward = 0.20

                elif delta_power <= 8:

                    reward = 0.00

                elif delta_power <= 15:

                    reward = -0.40

                else:

                    reward = -0.80

        # -------------------------------------------------------------
        # CASE B: IDLE GPU
        # -------------------------------------------------------------

        elif is_idle:

            # Lower power-cap actions receive higher rewards.

            normalized_action = (
                self.current_action + 1
            )

            reward = (
                1.0 /
                normalized_action
            )

            # Scale into a useful range.

            reward = (
                0.25 +
                0.75 * reward
            )

        # -------------------------------------------------------------
        # CASE C: NON-MEMORY COMPUTE
        # -------------------------------------------------------------

        elif not is_memory_bound:

            g = self._calculate_g(
                delta_power,
                delta_frequency
            )

            core_util = max(
                self.gpu_util / 100.0,
                0.20
            )

            reward = (
                g /
                core_util
            )

        # -------------------------------------------------------------
        # CASE D: MEMORY INTENSIVE
        # -------------------------------------------------------------

        else:

            g = self._calculate_g(
                delta_power,
                delta_frequency
            )

            memory_util = (
                self.mem_util /
                100.0
            )

            reward = (
                g *
                memory_util
            )

        # ============================================================
        # THERMAL SHAPING
        # ============================================================

        # This is our extension to the DRLCap-inspired reward.

        if self.temperature <= self.target_temperature:

            thermal_bonus = 0.10

        elif self.temperature <= self.warning_temperature:

            thermal_excess = (
                self.temperature -
                self.target_temperature
            )

            thermal_bonus = (
                0.10 -
                0.02 *
                thermal_excess
            )

        else:

            thermal_excess = (
                self.temperature -
                self.warning_temperature
            )

            thermal_bonus = (
                -0.15 -
                0.08 *
                thermal_excess
            )

        reward += thermal_bonus

        # ============================================================
        # ACTION OSCILLATION PENALTY
        # ============================================================

        action_change = abs(
            self.current_action -
            self.previous_action
        )

        if action_change > 1:

            reward -= (
                0.03 *
                (action_change - 1)
            )

        # ============================================================
        # CRITICAL TEMPERATURE
        # ============================================================

        if self.temperature >= self.critical_temperature:

            reward -= 1.0

        # ============================================================
        # REWARD CLIPPING
        # ============================================================

        reward = np.clip(
            reward,
            -1.5,
            1.5
        )

        self.last_delta_power = delta_power
        self.last_delta_frequency = delta_frequency
        self.last_reward = float(reward)

        return float(reward)

    # =================================================================
    # STEP
    # =================================================================

    def step(self, action_idx):

        # ============================================================
        # ROBUST ACTION VALIDATION
        # ============================================================

        try:

            action_idx = int(action_idx)

        except Exception:

            action_idx = self.previous_action

        action_idx = max(
            0,
            min(
                action_idx,
                self.n_actions - 1
            )
        )

        self.current_action = action_idx

        # ============================================================
        # SAVE PREVIOUS STATE
        # ============================================================

        self.previous_power = self.power
        self.previous_frequency = self.frequency

        # ============================================================
        # APPLY ACTION
        # ============================================================

        self.power_cap = float(
            self.power_caps[action_idx]
        )

        # ============================================================
        # WORKLOAD EVOLUTION
        # ============================================================

        self._update_workload()

        # ============================================================
        # GPU DYNAMICS
        # ============================================================

        self._update_frequency()

        self._update_utilization()

        self._update_power()

        self._update_temperature()

        # ============================================================
        # REWARD
        # ============================================================

        reward = self._calculate_reward()

        # ============================================================
        # PREVIOUS ACTION
        # ============================================================

        self.previous_action = self.current_action

        # ============================================================
        # STEP COUNT
        # ============================================================

        self.step_count += 1

        # ============================================================
        # TERMINATION
        # ============================================================

        done = False

        if self.step_count >= self.max_steps:

            done = True

        if self.temperature >= self.critical_temperature:

            done = True

        # ============================================================
        # NEXT STATE
        # ============================================================

        next_state = self._get_state()

        return (
            next_state,
            reward,
            done
        )

    # =================================================================
    # STATE
    # =================================================================

    def _get_state(self):

        return np.array(
            [
                # GPU utilization
                self.gpu_util / 100.0,

                # Memory utilization
                self.mem_util / 100.0,

                # Power
                self.power / self.max_power,

                # Frequency
                self.frequency / self.max_clock,

                # Temperature
                self.temperature / 100.0,

                # Workload intensity
                self.workload,

                # Power cap
                self.power_cap / self.max_power
            ],
            dtype=np.float32
        )

    # =================================================================
    # METRICS FOR LOGGING
    # =================================================================

    def get_metrics(self):

        return {
            "gpu_util": float(self.gpu_util),
            "mem_util": float(self.mem_util),
            "power": float(self.power),
            "frequency": float(self.frequency),
            "temperature": float(self.temperature),
            "workload": float(self.workload),
            "power_cap": float(self.power_cap),
            "delta_power": float(self.last_delta_power),
            "delta_frequency": float(self.last_delta_frequency),
            "reward": float(self.last_reward)
        }