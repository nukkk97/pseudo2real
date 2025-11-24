#DOMAINS="cooking;audio;transport;news;calendar;lists;iot;takeaway;play"
source /work/u3359154/miniconda3/bin/activate
conda activate task_vector
cd /work/u3359154/syn2real/SYN2REAL
#for pseudo in $(seq 0 15); do
#  CUDA_VISIBLE_DEVICES=0 python train_hyper_search.py \
#    --domains "$DOMAINS" \
#    --syn False \
#    --fold 0 \
#    --cluster 16 \
#    --current_pseudo "$pseudo"
#done

#for pseudo in $(seq 0 15); do
#  CUDA_VISIBLE_DEVICES=0 python train_hyper_search.py \
#    --domains "$DOMAINS" \
#    --syn True \
#    --fold 0 \
#    --cluster 16 \
#    --current_pseudo "$pseudo"
#done

# CUDA_VISIBLE_DEVICES=5 python train.py \
#  --domains $DOMAINS \
#  --syn True \
#  --fold 0 \
#  --cluster 1 \
#  --current_pseudo 0

# CUDA_VISIBLE_DEVICES=5 python train.py \
#  --domains $DOMAINS \
#  --syn False \
#  --fold 0 \
#  --cluster 1 \
#  --current_pseudo 0

DOMAINS=(general recommendation)

for domain in $DOMAINS; do
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --domains "$domain" \
    --syn True \
    --fold 0 \
    --cluster 1 \
    --current_pseudo 0 \
    --model_path /work/u3359154/syn2real/SYN2REAL/outputs/whisper_slurp_mixed_fold1
done
# DOMAINS="general;weather;qa;social;music;datetime;alarm;email;recommendation"

#for domain in $DOMAINS; do
#  python train.py \
#    --domains "$domain" \
#    --syn Mixed \
#    --fold 0 \
#    --cluster 1 \
#    --current_pseudo 0
#done