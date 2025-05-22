# Overcoming Sparsity Artifacts in Crosscoders to  Interpret Chat-Tuning

This repository contains the code for the **Overcoming Sparsity Artifacts in Crosscoders to  Interpret Chat-Tuning** paper.

 

## Requirements

Our code heavily relies on an adapted version of the `dictionary_learning` library. This code is separately provided. Install requirements with 
```bash
pip install -r requirements.txt
```

## Reproduce experiments

We cache model activations to disk. Our code assumes that you have around 4TB of storage per model available and that the environment variable `DATASTORE` points to it. The training scripts will log progress to [wandb](https://wandb.ai/). All models will be checkpointed to the `checkpoints` folder. The resulting plots will be generated in `$DATASTORE/results`.
The code requires that the environment variable `HF_HOME` points to a Hugging Face account or organization that you have write access to, as it will automatically upload multiple models and datasets to Hugging Face Hub during training and evaluation.

For Gemma 2 2b:
```bash
bash train_gemma2b.sh
```

For Llama 3.2 1b:
```bash
bash train_llama1b.sh
```

For Llama 3.1 8b:
```bash
bash train_llama8b.sh
```

Check out `notebooks/art.py` for generating the more complex plots.

## Code structure

The code that implements the actual crosscoders is found in our `dictionary_learning` fork.
This repository is organized into two main directories:

The folder [`scripts`](scripts/) contains the main execution scripts
- [`train_crosscoder.py`](scripts/train_crosscoder.py) - **Primary crosscoder training script**. Trains crosscoders on paired activations from base and chat models with support for various architectures (ReLU, batch-top-k) and normalization schemes.
- [`compute_scalers.py`](scripts/compute_scalers.py) - **Computes Latent Scalers using closed-form solution**. Calculates beta values for a given crosscoder.

- [`collect_activations.py`](scripts/collect_activations.py) - Caches model activations for training
- [`collect_dictionary_activations.py`](scripts/collect_dictionary_activations.py) - Collects activations through trained dictionaries
- [`collect_activating_examples.py`](scripts/collect_activating_examples.py) - Gathers max-activating examples for analysis. Required for demo.

The [`tools`](tools/) folder contains various utility functions. The [`steering_app`](steering_app/) folder contains a streamlit app to generate steered outputs.

With the [`notebooks/dashboard-and-demo.ipynb`](notebooks/dashboard-and-demo.ipynb) notebook you can explore the crosscoders and their latents.
