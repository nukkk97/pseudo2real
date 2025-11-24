# - Fold 0: yoruba, ijaw, afrikaans, idoma, setswana (total samples: 23,365)
# - Fold 1: igbo, swahili, hausa, zulu, twi (total samples: 24,067)

#DOMAINS=("yoruba" "ijaw" "afrikaans" "idoma" "setswana")
DOMAINS="yoruba;ijaw;afrikaans;idoma;setswana"

for domain in $DOMAINS; do
    # Train on synthetic data
    CUDA_VISIBLE_DEVICES=0 python train_afris_custom.py \
      --domains "$domain" \
      --model_path "openai/whisper-tiny" \
      --synth_text "text_whisper-tiny" \
      --configs "/work/u3359154/syn2real/SYN2REAL/configs/whisper_tiny.yaml" \
      --syn True \
      --fold 0 \
      --cluster 1 \
      --current_pseudo 0

    CUDA_VISIBLE_DEVICES=0 python train_afris_custom.py \
      --domains "$domain" \
      --model_path "openai/whisper-tiny" \
      --synth_text "text_whisper-tiny" \
      --configs "/work/u3359154/syn2real/SYN2REAL/configs/whisper_tiny.yaml" \
      --syn False \
      --fold 0 \
      --cluster 1 \
      --current_pseudo 0
done
#DOMAINS="igbo;swahili;hausa;zulu;twi"
#for domain in $DOMAINS; do
#  python train_afris.py \
#    --domains $DOMAINS \
#    --syn Mixed \
#    --fold 1 \
#    --cluster 1 \
#    --current_pseudo 0
#done
#DOMAINS="igbo swahili hausa zulu twi"

#for domain in "${DOMAINS[@]}"; do
# CUDA_VISIBLE_DEVICES=0 python train_afris.py \
#   --domains "$domain" \
#   --syn False \
#   --fold 0 \
#   --cluster 1 \
#   --current_pseudo 0
# CUDA_VISIBLE_DEVICES=0 python train_afris.py \
#   --domains "$domain" \
#   --syn True \
#   --fold 0 \
#   --cluster 1 \
#   --current_pseudo 0
#done

# DOMAINS="general;weather;qa;social;music;datetime;alarm;email;recommendation"