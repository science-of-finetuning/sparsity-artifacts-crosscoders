#! /bin/bash

set -x

SUBSET_SIZE=30000

BASE_DATASET=jkminder/fineweb-1m-sample
CHAT_DATASET=jkminder/lmsys-chat-1m-gemma-formatted
SPLIT=validation
BATCH_SIZE=32

FLAGS="--crosscoder-path Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04 \
    --base-dataset $BASE_DATASET \
    --chat-dataset $CHAT_DATASET \
    --dataset-split $SPLIT \
    --batch-size $BATCH_SIZE \
    --layer 13 \
    --subset-size $SUBSET_SIZE \
    --base-model google/gemma-2-2b \
    --instruct-model google/gemma-2-2b-it \
    --device cuda \
    --results-dir $DATASTORE/results/latent-stats"

additional_flags=$@

python scripts/compute_latent_statistics.py $FLAGS $additional_flags