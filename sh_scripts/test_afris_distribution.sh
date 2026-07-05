#!/bin/bash
# Path to test.py
SCRIPT="/work/u3359154/pseudo2real/test_afris_distribution.py"

# Pre-calculated source domain vector
VECTOR="/work/u3359154/pseudo2real/outputs/whisper_afris_diff_average_tiny2tiny.safetensors"

# Path to target domain synth models directory
MODEL_DIR="/work/u3359154/pseudo2real/outputs/tiny2tiny"

# List of weights to try
WEIGHTS=(0.0)

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
            --output_name "/work/u3359154/pseudo2real/outputs/distribution_tinytiny.txt"
    done
done