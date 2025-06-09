# Compute activations
bash scripts/llama_1b/compute-activations.sh --split train --dataset chat
bash scripts/llama_1b/compute-activations.sh --split val --dataset chat
bash scripts/llama_1b/compute-activations.sh --split train --dataset fineweb
bash scripts/llama_1b/compute-activations.sh --split val --dataset fineweb

bash scripts/llama_1b/train_crosscoder.sh --mu 0.036
bash scripts/llama_1b/train_crosscoder.sh --type batch-top-k --norm-init-scale 1.0 --k 100

# Analyze crosscoder
# This will upload the crosscoder to the huggingface hub. Use --no-upload to skip this.
bash run/llama_1b/pipeline.sh --crosscoder checkpoints/Meta-Llama-3.2-1B-L8-mu3.6e-02-lr1e-04-local-shuffling-CrosscoderLoss/model_final.pt 
bash run/llama_1b/pipeline.sh --crosscoder checkpoints/Meta-Llama-3.2-1B-L8-k100-lr1e-04-local-shuffling-Crosscoder/model_final.pt 
