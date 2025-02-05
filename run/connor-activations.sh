#! /bin/bash

set -x

# Define datasets and other constants
CHAT_DATASET=science-of-finetuning/lmsys-chat-1m-gemma-formatted
ACTIVATION_STORE_DIR=/workspace/data/activations
BATCH_SIZE=128
CHAT_MODEL=google/gemma-2-2b-it
BASE_MODEL=google/gemma-2-2b
TEXT_COLUMN=text_base_format

# Initialize variables for command-line arguments
SPLIT_ARG=""
DATASET_ARG=""
# Parse the command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --split)
            if [ -n "$2" ] && [[ "$2" != --* ]]; then
                SPLIT_ARG="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --split <train|val>"
            exit 1
            ;;
    esac
done

# Validate that both arguments were supplied
if [ -z "$SPLIT_ARG" ] ; then
    echo "Usage: $0 --split <train|val>"
    exit 1
fi

# Validate split argument and set corresponding values
if [ "$SPLIT_ARG" == "train" ]; then
    SPLIT="train"
    N_TOKS=25_000_000
elif [ "$SPLIT_ARG" == "val" ]; then
    SPLIT="validation"
    N_TOKS=5_000_000
else
    echo "Error: --split must be either 'train' or 'val'"
    exit 1
fi

DATASET=$CHAT_DATASET

# Build common flags using the updated variables
COMMON_FLAGS="--dtype float32 \
--disable-multiprocessing \
--store-tokens \
--text-column $TEXT_COLUMN \
--batch-size $BATCH_SIZE \
--layers 13 \
--dataset $DATASET \
--dataset-split $SPLIT \
--activation-store-dir $ACTIVATION_STORE_DIR \
--max-tokens $N_TOKS"

# Run activation collection for both base and chat models
python scripts/collect_activations.py $COMMON_FLAGS --model $BASE_MODEL
python scripts/collect_activations.py $COMMON_FLAGS --model $CHAT_MODEL