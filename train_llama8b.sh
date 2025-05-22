# Compute activations
bash scripts/llama_8b/compute-activations.sh --split train --dataset chat
bash scripts/llama_8b/compute-activations.sh --split val --dataset chat
bash scripts/llama_8b/compute-activations.sh --split train --dataset fineweb
bash scripts/llama_8b/compute-activations.sh --split val --dataset fineweb

# Train crosscoder
bash scripts/llama_8b/train_crosscoder.sh --mu 0.021
bash scripts/llama_8b/train_crosscoder.sh --type batch-top-k --norm-init-scale 1.0 --k 100

# Analyze crosscoder
# This will upload the crosscoder to the huggingface hub. Use --no-upload to skip this.
bash run/llama_8b/pipeline.sh --crosscoder checkpoints/Meta-Llama-3.1-8B-L16-mu2.1e-02-lr1e-04-local-shuffling-CrosscoderLoss/model_final.pt --num-samples-betas 10_000_000
bash run/llama_8b/pipeline.sh --crosscoder checkpoints/Meta-Llama-3.1-8B-L16-k200-lr1e-04-local-shuffling-Crosscoder/model_final.pt --num-samples-betas 10_000_000