# Quick GPU and configuration smoke test for TensorFlow.
# Purpose:
# - Detects available GPUs and verifies that TensorFlow can access them.
#
# Notes:
# - This script is meant as a lightweight environment sanity check.
# - Useful before running full training to confirm GPU + config setup.

import tensorflow as tf

print("\n=== GPU Detection ===")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for i, g in enumerate(gpus):
        print(f"Available GPU {i}: {g.name}")
        print(f"  - Details: {tf.config.experimental.get_device_details(g)}")
    print(f"[OK] {len(gpus)} Physical GPU(s) detected.\n")
else:
    print("[WARN] No GPU detected. Using CPU.\n")
