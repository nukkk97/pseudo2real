#!/bin/bash
# Path to test.py
SCRIPT="/work/u3359154/pseudo2real/test.py"

# Pre-calculated source domain vector
VECTOR="/work/u3359154/pseudo2real/outputs/16clusters_fold1/synth2real_fold-0_LOO-exclude0.safetensors"

# Path to target domain synth models directory
MODEL_DIR="/work/u3359154/pseudo2real/outputs/test"

# List of weights to try
WEIGHTS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Loop through each model directory
for MODEL_PATH in "$MODEL_DIR"/*; do
    MODEL_NAME=$(basename "$MODEL_PATH")

    # Extract domain name (everything after 'small_')
    DOMAIN=${MODEL_NAME#*_continue_}

    for WEIGHT in "${WEIGHTS[@]}"; do
        echo "Running inference for domain: $DOMAIN with weight: $WEIGHT"
        python "$SCRIPT" \
            --model_target_syn "$MODEL_PATH" \
            --model_vector "$VECTOR" \
            --domain "$DOMAIN" \
            --weight "$WEIGHT" \
            --output_name "result_cluster16.txt"
    done
done