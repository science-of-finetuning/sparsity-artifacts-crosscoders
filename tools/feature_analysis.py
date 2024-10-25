import torch as th
from nnsight import NNsight
from dataclasses import dataclass
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

th.set_float32_matmul_precision('high')


@dataclass
class Features:
    base: th.Tensor
    instruction: th.Tensor
    def __init__(self, base: th.Tensor = None, instruction: th.Tensor = None, features: th.Tensor = None):
        if features is not None:
            self.base = features[..., 0, :]
            self.instruction = features[..., 1, :]
        elif base is not None and instruction is not None:
            self.base = base
            self.instruction = instruction

    def to(self, device):
        self.base = self.base.to(device)
        self.instruction = self.instruction.to(device)
        return self
    
    def __del__(self):
        del self.base
        del self.instruction
        th.cuda.empty_cache()
@th.no_grad()
def get_activations(batch, model, layer):
    nnmodel = NNsight(model)
    with nnmodel.trace(batch["input_ids"].to(model.device), attention_mask=batch["attention_mask"].to(model.device)):
        activations = nnmodel.model.layers[layer].output[0].save()
        nnmodel.model.layers[layer].output.stop()
    return activations.value

@th.no_grad()
def get_features(batch, base_model, instruction_model, ae, layer):
    base_activations = get_activations(batch, base_model, layer)
    instruction_activations = get_activations(batch, instruction_model, layer)
    activations = th.stack([base_activations, instruction_activations], dim=-2).to(th.float32)
    batch_size = base_activations.shape[0]
    activations = activations.view(-1, activations.shape[-2], activations.shape[-1])
    features = ae.encode(activations, no_sum=True)
    # rescale features by decoder column norms
    features = features * ae.decoder.weight.norm(dim=2).unsqueeze(0)
    return Features(features=features.view(batch_size, -1, features.shape[-2], features.shape[-1]))

def filter_stack_features(features, attention_mask):
    # features: (batch_size, seq_len, n_layers, dict_size)
    # attention_mask: (batch_size, seq_len)
    base_features = features.base.view(-1, features.base.shape[-1])[attention_mask.view(-1).bool()]
    instruction_features = features.instruction.view(-1, features.instruction.shape[-1])[attention_mask.view(-1).bool()]
    return Features(base_features, instruction_features)


@dataclass
class FeatureStatistics:
    base_avg_activation: th.Tensor
    instruction_avg_activation: th.Tensor
    base_non_zero_counts: th.Tensor
    instruction_non_zero_counts: th.Tensor
    both_non_zero_counts: th.Tensor
    total_tokens: int
    relative_activation_diff: th.Tensor

def feature_statistics(dataset, tokenizer, base_model, instruction_model, ae, layer, batch_size=128, non_zero_threshold=1e-8):
    base_avg_activation = th.zeros(ae.dict_size, device="cuda")
    instruction_avg_activation = th.zeros(ae.dict_size, device="cuda")
    base_non_zero_counts = th.zeros(ae.dict_size, device="cuda", dtype=th.int32)
    instruction_non_zero_counts = th.zeros(ae.dict_size, device="cuda", dtype=th.int32)
    both_non_zero_counts = th.zeros(ae.dict_size, device="cuda", dtype=th.int32)
    activation_diff = th.zeros(ae.dict_size, device="cuda")
    total_tokens = 0
    for batch in tqdm(DataLoader(dataset, batch_size=batch_size)):
        tokens = tokenizer(batch["text"], return_tensors="pt", max_length=1024, truncation=True, padding=True)
        features = get_features(tokens, base_model, instruction_model, ae, layer)
        features = filter_stack_features(features, tokens["attention_mask"])
        base_avg_activation += features.base.sum(dim=0)
        instruction_avg_activation += features.instruction.sum(dim=0)
        base_non_zero_counts += (features.base > non_zero_threshold).sum(dim=0)
        instruction_non_zero_counts += (features.instruction > non_zero_threshold).sum(dim=0)
        both_non_zero_counts += ((features.base > non_zero_threshold) & (features.instruction > non_zero_threshold)).sum(dim=0)
        activation_diff += (features.base - features.instruction).sum(dim=0)
        total_tokens += tokens["attention_mask"].sum()
    return FeatureStatistics(base_avg_activation / total_tokens, instruction_avg_activation / total_tokens, base_non_zero_counts, instruction_non_zero_counts, both_non_zero_counts, total_tokens, activation_diff / total_tokens)
