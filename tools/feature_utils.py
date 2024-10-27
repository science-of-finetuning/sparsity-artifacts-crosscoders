import torch as th


def filter_dead_features(features, other_tensors, non_zero_threshold=1e-8):
    mask = th.stack([feature > non_zero_threshold for feature in features]).all(dim=0)
    return [feature[mask] for feature in features], [tensor[mask] for tensor in other_tensors]


def dead_feature_indices(combined_feature_statistics, rescaled=False, non_zero_threshold=1e-8):
    if rescaled:
        mask = th.stack([statistic.rescaled.joint.non_zero_counts > non_zero_threshold for statistic in combined_feature_statistics]).all(dim=0)
    else:
        mask = th.stack([statistic.normal.joint.non_zero_counts > non_zero_threshold for statistic in combined_feature_statistics]).all(dim=0)
    return mask.logical_not().nonzero().flatten().tolist()

def mask_to_indices(mask):
    return th.nonzero(mask).squeeze().tolist()

def remove_dead_and_filter(tensor, dead_indices, filter_indices):
    dead_mask = th.zeros(len(tensor)).bool()
    dead_mask[dead_indices] = True
    filter_mask = th.zeros(len(tensor)).bool()
    filter_mask[filter_indices] = True
    filtered_mask = ~dead_mask & filter_mask
    return tensor[filtered_mask]


    