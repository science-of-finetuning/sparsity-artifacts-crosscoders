# Compute activations
bash scripts/gemma2_2b/compute-activations.sh --split train --dataset chat
bash scripts/gemma2_2b/compute-activations.sh --split val --dataset chat
bash scripts/gemma2_2b/compute-activations.sh --split train --dataset fineweb
bash scripts/gemma2_2b/compute-activations.sh --split val --dataset fineweb

# Train crosscoder
bash scripts/gemma2_2b/train_crosscoder.sh --mu 0.041
bash scripts/gemma2_2b/train_crosscoder.sh --type batch-top-k --norm-init-scale 1.0 --k 100

# Analyze crosscoder
# This will upload the crosscoder to the huggingface hub. Use --no-upload to skip this.
bash run/gemma2_2b/pipeline.sh --crosscoder checkpoints/gemma-2-2b-L13-mu4.1e-02-lr1e-04-local-shuffling-CrosscoderLoss/model_final.pt --run-kl-experiment --compute-latent-stats
bash run/gemma2_2b/pipeline.sh --crosscoder checkpoints/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-Crosscoder/model_final.pt --run-kl-experiment --compute-latent-stats