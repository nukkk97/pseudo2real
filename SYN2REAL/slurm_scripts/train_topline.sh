DOMAINS="cooking audio transport news calendar lists iot takeaway play"

for domain in $DOMAINS; do
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --domains "$domain" \
    --syn True \
    --fold 0 \
    --cluster 1 \
    --current_pseudo 0
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --domains "$domain" \
    --syn False \
    --fold 0 \
    --cluster 1 \
    --current_pseudo 0
done