from real_gpu_control import (
    initialize_gpu_safety,
    set_power_limit,
    restore_gpu
)

import time


try:

    # --------------------------------------------------------
    # INITIALIZE SAFETY SYSTEM
    # --------------------------------------------------------

    initialize_gpu_safety()

    # --------------------------------------------------------
    # MANUAL TEST
    # --------------------------------------------------------

    print("\nSetting test power limit...")

    set_power_limit(250)

    print("\nPower limit applied.")

    time.sleep(10)

finally:

    # --------------------------------------------------------
    # ALWAYS RESTORE
    # --------------------------------------------------------

    restore_gpu()