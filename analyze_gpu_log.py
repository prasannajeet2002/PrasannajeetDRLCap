import pandas as pd
import numpy as np


FILE = "real_gpu_experiments/4070ti_monitor.csv"


df = pd.read_csv(FILE)


# ------------------------------------------------------------
# Convert timestamp into elapsed time
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


df["time_s"] = (
    df["timestamp"]
    - df["timestamp"].iloc[0]
).dt.total_seconds()


# ------------------------------------------------------------
# Energy integration
#
# E = integral(P dt)
# ------------------------------------------------------------

time_values = df["time_s"].values

power_values = (
    df["power_draw_W"]
    .values
)


energy_joules = np.trapezoid(
    power_values,
    time_values
)


execution_time = (
    time_values[-1]
    - time_values[0]
)


average_power = (
    power_values.mean()
)


edp = (
    energy_joules
    * execution_time
)


print("\n====================================")
print("GPU EXPERIMENT RESULT")
print("====================================")

print(
    "Execution time:",
    execution_time,
    "seconds"
)

print(
    "Average power:",
    average_power,
    "W"
)

print(
    "Energy:",
    energy_joules,
    "J"
)

print(
    "Energy:",
    energy_joules / 1000,
    "kJ"
)

print(
    "EDP:",
    edp,
    "J.s"
)

print(
    "Average GPU utilization:",
    df["gpu_util_percent"].mean(),
    "%"
)

print(
    "Average temperature:",
    df["temperature_C"].mean(),
    "C"
)

print(
    "Maximum temperature:",
    df["temperature_C"].max(),
    "C"
)
