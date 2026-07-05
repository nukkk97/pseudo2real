# - Fold 0: yoruba, ijaw, afrikaans, idoma, setswana (total samples: 23,365)
# - Fold 1: igbo, swahili, hausa, zulu, twi (total samples: 24,067)

#DOMAINS=("yoruba" "ijaw" "afrikaans" "idoma" "setswana")
#!/bin/bash

#!/bin/bash

DOMAINS="yoruba;ijaw;afrikaans;idoma;setswana"

# Cluster 2 → pseudo 0..1
for pseudo in {0..1}; do
  cluster=2

  # syn=True
  CUDA_VISIBLE_DEVICES=0 python train_afris_custom.py \
    --domains "$DOMAINS" \
    --model_path "openai/whisper-small" \
    --synth_text "text_whisper-medium" \
    --syn True \
    --fold 0 \
    --cluster "$cluster" \
    --current_pseudo "$pseudo"

  # syn=False
  CUDA_VISIBLE_DEVICES=0 python train_afris_custom.py \
    --domains "$DOMAINS" \
    --model_path "openai/whisper-small" \
    --synth_text "text_whisper-medium" \
    --syn False \
    --fold 0 \
    --cluster "$cluster" \
    --current_pseudo "$pseudo"
done
