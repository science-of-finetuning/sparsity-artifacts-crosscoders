import torch as th


def filter_dead_latents(latents, other_tensors, non_zero_threshold=1e-8):
    mask = th.stack([latent > non_zero_threshold for latent in latents]).all(dim=0)
    return [latent[mask] for latent in latents], [
        tensor[mask] for tensor in other_tensors
    ]


def dead_latent_indices(combined_latent_statistics, rescaled=False):
    if rescaled:
        mask = th.stack(
            [
                statistic.rescaled.joint.non_zero_counts > 0
                for statistic in combined_latent_statistics
            ]
        ).all(dim=0)
    else:
        mask = th.stack(
            [
                statistic.normal.joint.non_zero_counts > 0
                for statistic in combined_latent_statistics
            ]
        ).all(dim=0)
    return mask.logical_not().nonzero().flatten().tolist()


def mask_to_indices(mask):
    return th.nonzero(mask).squeeze().tolist()

def indices_to_mask(indices, length):
    mask = th.zeros(length).bool()
    mask[indices] = True
    return mask

def remove_dead_and_filter(tensor, dead_indices, filter_indices):
    dead_mask = th.zeros(len(tensor)).bool()
    dead_mask[dead_indices] = True
    filter_mask = th.zeros(len(tensor)).bool()
    filter_mask[filter_indices] = True
    filtered_mask = ~dead_mask & filter_mask
    return tensor[filtered_mask]
