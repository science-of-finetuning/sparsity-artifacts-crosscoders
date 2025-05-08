#! /bin/bash

set -x

SPLIT=train
BATCH_SIZE=2048
WORKERS=32
# Model configuration
ACTIVATION_DIR="$DATASTORE/activations/"
LAYER=16
CHAT_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
BASE_MODEL=meta-llama/Meta-Llama-3.1-8B
DEVICE="cuda"
LR=1e-4
MU=0.041
TEXT_COLUMN="text_llama3"

# Parse command line arguments to check for custom mu value
custom_mu=false
custom_lr=false
for arg in "$@"; do
    if [[ $arg == --mu* ]]; then
        custom_mu=true
    fi
    if [[ $arg == --lr* ]]; then
        custom_lr=true
    fi
done

# Build flags string
FLAGS="--activation-store-dir $ACTIVATION_DIR \
--batch-size $BATCH_SIZE \
--layer $LAYER \
--base-model $BASE_MODEL \
--chat-model $CHAT_MODEL \
--same-init-for-all-layers \
--init-with-transpose \
--validate-every-n-steps 20_000 \
--epochs 2 \
--local-shuffling \
--seed 42 \
--num-samples 100_000_000 \
--text-column $TEXT_COLUMN"

# Only add default mu if not provided in command line arguments
if [ "$custom_mu" = false ]; then
    FLAGS="$FLAGS --mu $MU"
fi
if [ "$custom_lr" = false ]; then
    FLAGS="$FLAGS --lr $LR"
fi
additional_flags=$@

python scripts/train_crosscoder.py $FLAGS $additional_flags
