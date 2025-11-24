DOMAINS="general;weather;qa;social;music;datetime;alarm;email;recommendation"
CLUSTER=16
source /work/u3359154/miniconda3/bin/activate
conda activate task_vector
cd /work/u3359154/syn2real/SYN2REAL
#for PSEUDO in $(seq 0 15); do
#  echo "▶️ Running SYNTHETIC cluster $PSEUDO"
#  python train_random.py \
#    --domains "$DOMAINS" \
#    --syn True \
#    --cluster $CLUSTER \
#    --current_pseudo $PSEUDO
#
#  echo "▶️ Running REAL cluster $PSEUDO"
#  python train_random.py \
#    --domains "$DOMAINS" \
#    --syn False \
#    --cluster $CLUSTER \
#    --current_pseudo $PSEUDO
#done
for PSEUDO in $(seq 0 3); do
  echo "▶️ Running Synth cluster $PSEUDO"
  python train_random.py \
    --domains "$DOMAINS" \
    --syn True \
    --cluster $CLUSTER \
    --current_pseudo $PSEUDO
done