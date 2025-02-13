#! /bin/bash

set -x

# Define datasets and other constants
CHAT_DATASET=science-of-finetuning/lmsys-chat-1m-chat-formatted
FINEWEB_DATASET=science-of-finetuning/fineweb-1m-sample
ACTIVATION_STORE_DIR=/workspace/data/activations
BATCH_SIZE=64
CHAT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
BASE_MODEL=Qwen/Qwen2.5-1.5B
LAYERS=14

# Initialize variables for command-line arguments
SPLIT_ARG=""
DATASET_ARG=""
OTHER_FLAGS=""
CHAT_ONLY=false
BASE_ONLY=false

# Parse the command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --split)
            SPLIT_ARG="$2"
            shift 2
            ;;
        --dataset)
            DATASET_ARG="$2"
            shift 2
            ;;
        --chat-only)
            CHAT_ONLY=true
            shift
            ;;
        --base-only)
            BASE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --split <train|val> --dataset <chat|fineweb> [--chat-only] [--base-only]"
            exit 1
            ;;
    esac
done

# Validate that both arguments were supplied
if [ -z "$SPLIT_ARG" ] || [ -z "$DATASET_ARG" ]; then
    echo "Usage: $0 --split <train|val> --dataset <chat|fineweb>"
    exit 1
fi

# Validate split argument and set corresponding values
if [ "$SPLIT_ARG" == "train" ]; then
    SPLIT="train"
    N_TOKS=100_000_000
elif [ "$SPLIT_ARG" == "val" ]; then
    SPLIT="validation"
    N_TOKS=5_000_000
else
    echo "Error: --split must be either 'train' or 'val'"
    exit 1
fi

# Validate dataset argument and choose dataset accordingly
if [ "$DATASET_ARG" == "chat" ]; then
    DATASET=$CHAT_DATASET
    BATCH_SIZE=48
elif [ "$DATASET_ARG" == "fineweb" ]; then
    DATASET=$FINEWEB_DATASET
    OTHER_FLAGS="--text-column text"
else
    echo "Error: --dataset must be either 'chat' or 'fineweb'"
    exit 1
fi

# Build common flags using the updated variables
COMMON_FLAGS="--wandb --overwrite --dtype float32 \
--disable-multiprocessing \
--store-tokens \
--batch-size $BATCH_SIZE \
--layers $LAYERS \
--dataset $DATASET \
--dataset-split $SPLIT \
--activation-store-dir $ACTIVATION_STORE_DIR \
--max-tokens $N_TOKS"

# Run activation collection based on flags
if [ "$BASE_ONLY" = true ] && [ "$CHAT_ONLY" = true ]; then
    echo "Error: Cannot specify both --chat-only and --base-only"
    exit 1
fi

if [ "$BASE_ONLY" = true ]; then
    python scripts/collect_activations.py $COMMON_FLAGS --model $BASE_MODEL $OTHER_FLAGS
elif [ "$CHAT_ONLY" = true ]; then
    python scripts/collect_activations.py $COMMON_FLAGS --model $CHAT_MODEL $OTHER_FLAGS
else
    # Run both models if no specific flag is set
    python scripts/collect_activations.py $COMMON_FLAGS --model $BASE_MODEL $OTHER_FLAGS
    python scripts/collect_activations.py $COMMON_FLAGS --model $CHAT_MODEL $OTHER_FLAGS
fi
