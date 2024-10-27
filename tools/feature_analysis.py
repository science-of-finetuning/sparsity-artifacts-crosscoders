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
        assert features is not None or (base is not None and instruction is not None)
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
    rescaled_features = features * ae.decoder.weight.norm(dim=2).unsqueeze(0)
    return Features(features=rescaled_features.view(batch_size, -1, rescaled_features.shape[-2], rescaled_features.shape[-1])), Features(features=features.view(batch_size, -1, features.shape[-2], features.shape[-1]))

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
    abs_activation_diff: th.Tensor
    rel_activation_diff: th.Tensor
    either_non_zero_counts: th.Tensor
    is_normalized: bool = False

    def normalize(self):
        if self.is_normalized:
            return self
        self.base_avg_activation /= self.total_tokens
        self.instruction_avg_activation /= self.total_tokens
        self.abs_activation_diff /= self.total_tokens
        self.rel_activation_diff /= self.either_non_zero_counts
        self.is_normalized = True
        return self

    def combine(self, other):
        self.base_avg_activation += other.base_avg_activation
        self.instruction_avg_activation += other.instruction_avg_activation
        self.abs_activation_diff += other.abs_activation_diff
        self.rel_activation_diff += other.rel_activation_diff
        self.base_non_zero_counts += other.base_non_zero_counts
        self.instruction_non_zero_counts += other.instruction_non_zero_counts
        self.both_non_zero_counts += other.both_non_zero_counts
        self.total_tokens += other.total_tokens
        self.either_non_zero_counts += other.either_non_zero_counts
        return self

@dataclass
class CombinedFeatureStatistics:
    rescaled: FeatureStatistics
    normal: FeatureStatistics
    
    def normalize(self):
        self.rescaled.normalize()
        self.normal.normalize()
        return self

def compute_statistics(features, total_tokens, non_zero_threshold=1e-8):
    base_avg_activation = features.base.sum(dim=0)  
    instruction_avg_activation = features.instruction.sum(dim=0) 
    base_non_zero_counts = (features.base > non_zero_threshold).sum(dim=0)
    instruction_non_zero_counts = (features.instruction > non_zero_threshold).sum(dim=0)
    both_non_zero_counts = ((features.base > non_zero_threshold) & (features.instruction > non_zero_threshold)).sum(dim=0)
    activation_diff = (features.base - features.instruction).sum(dim=0) 
    rel_activation_diff = (((features.base - features.instruction) / (th.stack([features.base, features.instruction], dim=0).max(dim=0).values + 1e-8) + 1) / 2)
    # only consider features that are non-zero in either base or instruction -> set others to 0
    rel_activation_diff = rel_activation_diff * ((features.base > non_zero_threshold) | (features.instruction > non_zero_threshold)).float()
    rel_activation_diff = rel_activation_diff.sum(dim=0)
    either_non_zero_counts = ((features.base > non_zero_threshold) | (features.instruction > non_zero_threshold)).sum(dim=0)
    return FeatureStatistics(
        base_avg_activation,
        instruction_avg_activation,
        base_non_zero_counts,
        instruction_non_zero_counts,
        both_non_zero_counts,
        total_tokens,
        activation_diff,
        rel_activation_diff,
        either_non_zero_counts
    )

def feature_statistics(dataset, tokenizer, base_model, instruction_model, ae, layer, batch_size=128, non_zero_threshold=1e-8):
    batch_idx = 0
    rescaled_stats = None
    normal_stats = None

    for batch in tqdm(DataLoader(dataset, batch_size=batch_size)):
        tokens = tokenizer(batch["text"], return_tensors="pt", max_length=1024, truncation=True, padding=True)
        batch_rescaled_features, batch_normal_features = get_features(tokens, base_model, instruction_model, ae, layer)
        batch_rescaled_features = filter_stack_features(batch_rescaled_features, tokens["attention_mask"])
        batch_normal_features = filter_stack_features(batch_normal_features, tokens["attention_mask"])
        rescaled_stats_batch = compute_statistics(batch_rescaled_features, tokens["attention_mask"].sum(), non_zero_threshold)
        normal_stats_batch = compute_statistics(batch_normal_features, tokens["attention_mask"].sum(), non_zero_threshold)

        if batch_idx == 0:
            rescaled_stats = rescaled_stats_batch
            normal_stats = normal_stats_batch
        else:
            rescaled_stats.combine(rescaled_stats_batch)
            normal_stats.combine(normal_stats_batch)

        batch_idx += 1


    rescaled_stats.normalize()
    normal_stats.normalize()

    return CombinedFeatureStatistics(rescaled_stats, normal_stats)
