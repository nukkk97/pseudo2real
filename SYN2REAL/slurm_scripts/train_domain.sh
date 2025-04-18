DOMAINS="cooking audio transport news calender lists iot takeaway play"

for domain in $DOMAINS; do
  CUDA_VISIBLE_DEVICES=1 python train.py \
    --domains "$domain" \
    --syn True \
    --fold 0 \
    --cluster 1 \
    --current_pseudo 0
done