#! /bin/bash

set -x

SPLIT=train
BATCH_SIZE=128
WORKERS=32
FLAGS="--crosscoder-path Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04 --activation-store-dir $DATASTORE/activations/ --dataset-split $SPLIT --batch-size $BATCH_SIZE --layer 13 --base-model gemma-2-2b --instruct-model gemma-2-2b-it --device cuda --num-workers $WORKERS"
additional_flags=$@

python scripts/compute_scalers.py $FLAGS $additional_flags 