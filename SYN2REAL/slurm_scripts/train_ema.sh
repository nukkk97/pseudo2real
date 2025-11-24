python train_ema.py \
  --domains "yoruba;ijaw;afrikaans;idoma;setswana" \
  --syn True \
  --fold 1 \
  --cluster 1 \
  --current_pseudo 0 \
  --model_path /work/u3359154/syn2real/SYN2REAL/outputs/whisper_afris_mixed_fold-1_cluster-0-of-1_tiny_to_tiny \
  --configs /work/u3359154/syn2real/SYN2REAL/configs/whisper_tiny.yaml \
  --ema True --ema_decay 0.9995 --ema_warmup_steps 150 --ema_update_every 1 --ema_ramp cosine