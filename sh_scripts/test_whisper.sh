SCRIPT="/work/u3359154/pseudo2real/test_afris.py"
MODEL_DIR="openai/whisper-medium"
DOMAINS="hausa;igbo;swahili;twi;zulu"

# Convert semicolon-separated string into an array
IFS=';' read -ra DOMAIN_ARRAY <<< "$DOMAINS"

for DOMAIN in "${DOMAIN_ARRAY[@]}"; do
    echo "Running for domain: $DOMAIN"
    python "$SCRIPT" \
        --model_path "$MODEL_DIR" \
        --domain "$DOMAIN" \
        --output_name "/work/u3359154/pseudo2real/outputs/whisper_medium.txt"
done