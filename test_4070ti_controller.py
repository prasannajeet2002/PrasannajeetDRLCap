from gpu_controller_4070ti import (
    initialize,
    get_gpu_info,
    set_power_limit,
    restore
)

import time


try:

    # --------------------------------------------------------
    # 1. Read and save original configuration
    # --------------------------------------------------------

    initialize()

    # --------------------------------------------------------
    # 2. Read current GPU state
    # --------------------------------------------------------

    print("\nCurrent GPU information:")

    info = get_gpu_info()

    print(info)

    # --------------------------------------------------------
    # 3. IMPORTANT:
    #
    # Replace this value only AFTER checking:
    #
    # nvidia-smi -i 1 -q -d POWER
    #
    # --------------------------------------------------------

    TEST_POWER_LIMIT = 200

    print(
        "\nTesting power limit:",
        TEST_POWER_LIMIT,
        "W"
    )

    set_power_limit(TEST_POWER_LIMIT)

    # --------------------------------------------------------
    # 4. Observe GPU
    # --------------------------------------------------------

    print(
        "\nPower limit applied."
    )

    time.sleep(10)

    # --------------------------------------------------------
    # 5. Read GPU again
    # --------------------------------------------------------

    info = get_gpu_info()

    print("\nGPU after test:")
    print(info)

finally:

    # --------------------------------------------------------
    # 6. ALWAYS RESTORE
    # --------------------------------------------------------

    restore()
