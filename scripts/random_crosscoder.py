import torch as th
import argparse
from pathlib import Path    
from dictionary_learning.cache import PairedActivationCache


from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer, CrossCoder
from dictionary_learning.trainers import CrossCoderTrainer
from dictionary_learning.training import trainSAE
import os

th.set_float32_matmul_precision('high')

def random_data(shape):
    return th.randn(*shape)

class RandomDataset(th.utils.data.Dataset):
    def __init__(self, shape, device):
        self.shape = shape
        self.device = device

    def __len__(self):
        return 10**8

    def __getitem__(self, idx):
        return random_data(self.shape)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-store-dir", type=str, default="activations")
    parser.add_argument("--base-model", type=str, default="gemma-2-2b")
    parser.add_argument("--instruct-model", type=str, default="gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--wandb-entity", type=str, default="jkminder")
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--mu", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-every-n-steps", type=int, default=10000)
    args = parser.parse_args()

    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)


    activation_store_dir = Path(args.activation_store_dir) 

    base_model_dir = activation_store_dir / args.base_model
    instruct_model_dir = activation_store_dir / args.instruct_model

    base_model_fineweb = base_model_dir / "fineweb"
    base_model_lmsys_chat = base_model_dir / "lmsys_chat"
    instruct_model_fineweb = instruct_model_dir / "fineweb"
    instruct_model_lmsys_chat = instruct_model_dir / "lmsys_chat"

    submodule_name = f"layer_{args.layer}_out"

    fineweb_cache = PairedActivationCache(base_model_fineweb / submodule_name, instruct_model_fineweb / submodule_name)
    device = "cuda" if th.cuda.is_available() else "cpu"
    shape = fineweb_cache[0].shape

    dataset = RandomDataset(shape, device)

    activation_dim = shape[1]
    dictionary_size = args.expansion_factor * activation_dim
    print(f"Training on device={device}.")
    trainer_cfg = {
        "trainer": CrossCoderTrainer,
        "dict_class": CrossCoder,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": 1e-3,
        "device": device,
        "warmup_steps": 1000,
        "layer": args.layer,
        "lm_name": f"{args.instruct_model}-{args.base_model}",
        "compile": True,
        "wandb_name": f"random-crosscoder-mu{args.mu}",
        "l1_penalty": args.mu,
        "dict_class_kwargs": {
            "same_init_for_all_layers": True,
            "norm_init_scale": 0.005,
            "init_with_transpose": True,
        }
    }

    validation_size = 10**6
    train_dataset, validation_dataset = th.utils.data.random_split(dataset, [len(dataset) - validation_size, validation_size])
    print(f"Training on {len(train_dataset)} token activations.")
    dataloader = th.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    validation_dataloader = th.utils.data.DataLoader(validation_dataset, batch_size=8192, shuffle=False, num_workers=args.workers, pin_memory=True)

    # train the sparse autoencoder (SAE)
    ae = trainSAE(
        data=dataloader, 
        trainer_configs=[trainer_cfg],
        validate_every_n_steps=None,
        validation_data=validation_dataloader,
        use_wandb=args.wandb_entity is not None,
        wandb_entity=args.wandb_entity,
        wandb_project="crosscoder",
        log_steps=50,
        save_steps=10000,
        save_dir="checkpoints",
    )
