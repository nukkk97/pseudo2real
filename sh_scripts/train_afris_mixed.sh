# - Fold 0: yoruba, ijaw, afrikaans, idoma, setswana (total samples: 23,365)
# - Fold 1: igbo, swahili, hausa, zulu, twi (total samples: 24,067)

DOMAINS="yoruba;ijaw;afrikaans;idoma;setswana"

for domain in $DOMAINS; do
 CUDA_VISIBLE_DEVICES=0 python train_afris_custom.py \
   --domains "$domain" \
   --model_path "openai/whisper-tiny" \
   --synth_text "text_whisper-tiny" \
   --syn Mixed \
   --fold 0 \
   --cluster 1 \
   --current_pseudo 0
done

