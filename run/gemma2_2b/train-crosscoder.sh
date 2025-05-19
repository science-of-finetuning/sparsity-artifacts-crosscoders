#! /bin/bash

set -x

SPLIT=train
BATCH_SIZE=2048
WORKERS=32
# Model configuration
ACTIVATION_DIR="$DATASTORE/activations/"
LAYER=13
BASE_MODEL="google/gemma-2-2b"
INSTRUCT_MODEL="google/gemma-2-2b-it"
DEVICE="cuda"
LR=1e-4
MU=0.041

# Parse command line arguments to check for custom mu value
custom_mu=false
for arg in "$@"; do
    if [[ $arg == --mu* ]]; then
        custom_mu=true
        break
    fi
done

# Build flags string
FLAGS="--activation-store-dir $ACTIVATION_DIR \
--batch-size $BATCH_SIZE \
--layer $LAYER \
--base-model $BASE_MODEL \
--chat-model $CHAT_MODEL \
--same-init-for-all-layers \
--lr $LR \
--init-with-transpose \
--validate-every-n-steps 20_000 \
--epochs 2 \
--local-shuffling \
--seed 42 \
--num-samples 100_000_000"

# Only add default mu if not provided in command line arguments
if [ "$custom_mu" = false ]; then
    FLAGS="$FLAGS --mu $MU"
fi

additional_flags=$@

python scripts/train_crosscoder.py $FLAGS $additional_flags