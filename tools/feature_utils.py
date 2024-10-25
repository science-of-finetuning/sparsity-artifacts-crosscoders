import torch as th


def filter_dead_features(*features, non_zero_threshold=1e-8):
    mask = th.stack([feature > non_zero_threshold for feature in features]).all(dim=0)
    return [feature[mask] for feature in features]
