# Overcoming Sparsity Artifacts in Crosscoders to  Interpret Chat-Tuning

This repository contains the code for the **Overcoming Sparsity Artifacts in Crosscoders to  Interpret Chat-Tuning** paper.

The trained models, along with statistics and maximally activating examples for each latent, are hosted at [our huggingface page](https://huggingface.co/science-of-finetuning). We also provide an interactive [Colab notebook](https://dub.sh/ccdm) and training logs in our [wandb](https://wandb.ai/jkminder/chat-crosscoders).

 

## Requirements

Our code heavily relies on an adapted version of the [`dictionary_learning`](https://github.com/jkminder/dictionary_learning) library. Install requirements with 
```bash
pip install -r requirements.txt
```

## Reproduce experiments

We cache model activations to disk. Our code assumes that you have around 4TB of storage per model available and that the environment variable `$DATASTORE` points to it. The training scripts will log progress to [wandb](https://wandb.ai/). All models will be checkpointed to the `checkpoints` folder. The resulting plots will be generated in `$DATASTORE/results`.

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

The code that implements the actual crosscoders is found in our [`dictionary_learning`](https://github.com/jkminder/dictionary_learning) fork.
This repository is organized into two main directories:

The folder [`scripts`](scripts/) contains the main execution scripts
- [`train_crosscoder.py`](scripts/train_crosscoder.py) - **Primary crosscoder training script**. Trains crosscoders on paired activations from base and chat models with support for various architectures (ReLU, batch-top-k) and normalization schemes.
- [`compute_scalers.py`](scripts/compute_scalers.py) - **Computes Latent Scalers using closed-form solution**. Calculates beta values for a given crosscoder.

- [`collect_activations.py`](scripts/collect_activations.py) - Caches model activations for training
- [`collect_dictionary_activations.py`](scripts/collect_dictionary_activations.py) - Collects activations through trained dictionaries
- [`collect_activating_examples.py`](scripts/collect_activating_examples.py) - Gathers max-activating examples for analysis. Required for demo.

The [`tools`](tools/) folder contains various utility functions. The [`steering_app`](steering_app/) folder contains a streamlit app to generate steered outputs.

With the [`notebooks/dashboard-and-demo.ipynb`](notebooks/dashboard-and-demo.ipynb) notebook you can explore the crosscoders and their latents.
