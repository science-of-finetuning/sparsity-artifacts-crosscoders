#!/bin/bash
set -x

# Initialize variables for --n-train and --n-val.
n_train=""
n_val=""
no_train=false
other_args=()

# Parse input arguments.
while (( "$#" )); do
    case "$1" in
        --n-train)
            if [ -n "$2" ] && [[ "$2" != --* ]]; then
                n_train="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --n-val)
            if [ -n "$2" ] && [[ "$2" != --* ]]; then
                n_val="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --no-train)
            no_train=true
            shift
            ;;
        *)
            other_args+=("$1")
            shift
            ;;
    esac
done

# Prepare the -N flag for compute and verify scripts if provided.
compute_n=()
if [ -n "$n_train" ]; then
    compute_n=(-N "$n_train")
fi

verify_n=()
if [ -n "$n_val" ]; then
    verify_n=(-N "$n_val")
fi

N_TRAIN=${n_train:-1000000}
N_VAL=${n_val:-1000000}

# Invoke the compute-scalers.sh script with -N set from --n-train.
if [ "$no_train" = false ]; then
    bash run/compute-scalers.sh -N "$N_TRAIN" "${other_args[@]}"  
fi

# Invoke the verify-scalers.sh script twice (for train and validation)
# with -N set from --n-val, and add the appropriate dataset split.
bash run/verify-scalers.sh -N "$N_VAL" "${other_args[@]}" --dataset-split train --train-n-offset 0 --train-num-samples "$N_TRAIN"
bash run/verify-scalers.sh -N "$N_VAL" "${other_args[@]}" --dataset-split validation --train-n-offset 0 --train-num-samples "$N_TRAIN"
