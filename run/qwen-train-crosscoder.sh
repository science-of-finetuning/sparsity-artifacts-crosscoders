#! /bin/bash

set -x

SPLIT=train
BATCH_SIZE=2048
WORKERS=32
# Model configuration
ACTIVATION_DIR="$DATASTORE/activations/"
LAYER=14
BASE_MODEL="Qwen/Qwen2.5-1.5B"
INSTRUCT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda"
LR=1e-4
MU=0.041

# Build flags string
FLAGS="--activation-store-dir $ACTIVATION_DIR \
--batch-size $BATCH_SIZE \
--layer $LAYER \
--base-model $BASE_MODEL \
--chat-model $INSTRUCT_MODEL \
--same-init-for-all-layers \
--lr $LR \
--mu $MU \
--init-with-transpose \
--validate-every-n-steps 10_000 \
--text-column text_qwen2_5"

additional_flags=$@

python scripts/train_crosscoder.py $FLAGS $additional_flags 