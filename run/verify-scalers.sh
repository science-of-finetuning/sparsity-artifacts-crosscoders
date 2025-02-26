#! /bin/bash

set -x

BATCH_SIZE=128
WORKERS=32
FLAGS="--max-activations $DATASTORE/max_activations_N50000000.pt \
      --chat-only-indices-path $DATASTORE/only_it_decoder_feature_indices.pt \
      --crosscoder-path Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04 \
      --activation-store-dir $DATASTORE/activations/ \
      --batch-size $BATCH_SIZE \
      --layer 13 \
      --base-model gemma-2-2b \
      --instruct-model gemma-2-2b-it \
      --device cuda \
      --num-workers $WORKERS"
additional_flags=$@

python scripts/verify_scalers.py $FLAGS $additional_flags 