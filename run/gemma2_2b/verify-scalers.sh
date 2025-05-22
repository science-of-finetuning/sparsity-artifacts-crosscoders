#! /bin/bash

set -x

SPLIT=validation
BATCH_SIZE=128
WORKERS=32

# Default values
CROSSCODER_PATH="$HF_NAME/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04"
LATENT_INDICES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --crosscoder-path)
      CROSSCODER_PATH="$2"
      shift 2
      ;;
    --latent-indices)
      LATENT_INDICES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      REMAINING_ARGS+=("$1")
      shift
      ;;
  esac
done

# Set default latent-indices path if not provided
if [ -z "$LATENT_INDICES" ]; then
  # Replace / with _ in crosscoder path for directory name
  INDICES_DIR=$(echo $CROSSCODER_PATH | tr '/' '_')
  LATENT_INDICES="$DATASTORE/latent_indices/${INDICES_DIR}/chat_only_indices.pt"
fi

FLAGS="--crosscoder-path $CROSSCODER_PATH \
      --latent-indices-path $LATENT_INDICES \
      --activation-store-dir $DATASTORE/activations/ \
      --dataset-split $SPLIT \
      --batch-size $BATCH_SIZE \
      --layer 13 \
      --base-model google/gemma-2-2b \
      --chat-model google/gemma-2-2b-it \
      --device cuda \
      --num-workers $WORKERS"
      
python scripts/verify_scalers.py $FLAGS "${REMAINING_ARGS[@]}"