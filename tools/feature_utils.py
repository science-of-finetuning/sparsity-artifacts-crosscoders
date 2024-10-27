import torch as th


def filter_dead_features(features, other_tensors, non_zero_threshold=1e-8):
    mask = th.stack([feature > non_zero_threshold for feature in features]).all(dim=0)
    return [feature[mask] for feature in features], [tensor[mask] for tensor in other_tensors]


def dead_feature_indices(features, non_zero_threshold=1e-8):
    mask = th.stack([feature > non_zero_threshold for feature in features]).all(dim=0)
    return mask.logical_not().nonzero().flatten().tolist()
