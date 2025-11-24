# - Fold 0: yoruba, ijaw, afrikaans, idoma, setswana (total samples: 23,365)
# - Fold 1: igbo, swahili, hausa, zulu, twi (total samples: 24,067)

DOMAINS=("igbo" "swahili" "hausa" "zulu" "twi")
#DOMAINS="yoruba;ijaw;afrikaans;idoma;setswana"

for domain in "${DOMAINS[@]}"; do
 CUDA_VISIBLE_DEVICES=0 python train_afris_costum.py \
   --domains "$domain" \
   --model_path "openai/whisper-small" \
   --synth_text "text_whisper-large-v2" \
   --syn True \
   --fold 0 \
   --cluster 1 \
   --current_pseudo 0 \
   --model_path "/work/u3359154/syn2real/SYN2REAL/outputs/whisper_afris_mixed_fold-0_cluster-0-of-1_base_to_tiny" 
done