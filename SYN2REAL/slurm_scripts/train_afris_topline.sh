# - Fold 0: yoruba, ijaw, afrikaans, idoma, setswana (total samples: 23,365)
# - Fold 1: igbo, swahili, hausa, zulu, twi (total samples: 24,067)

DOMAINS=("afrikaans" "idoma" "setswana")
#DOMAINS="yoruba;ijaw;afrikaans;idoma;setswana"

for domain in "${DOMAINS[@]}"; do
 CUDA_VISIBLE_DEVICES=0 python train_afris_custom.py \
   --domains "$domain" \
   --model_path "openai/whisper-small" \
   --synth_text "text_whisper-small" \
   --syn False \
   --fold 0 \
   --cluster 1 \
   --current_pseudo 0
 CUDA_VISIBLE_DEVICES=0 python train_afris_custom.py \
   --domains "$domain" \
   --model_path "openai/whisper-small" \
   --synth_text "text_whisper-small" \
   --syn True \
   --fold 0 \
   --cluster 1 \
   --current_pseudo 0
done