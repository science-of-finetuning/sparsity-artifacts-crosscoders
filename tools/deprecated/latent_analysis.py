import torch as th
from nnsight import NNsight
from dataclasses import dataclass
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots

from .latent_utils import (
    filter_dead_latents,
    mask_to_indices,
    remove_dead_and_filter,
    dead_latent_indices,
)


th.set_float32_matmul_precision("high")


@dataclass
class Latents:
    base: th.Tensor
    instruction: th.Tensor
    joint: th.Tensor

    def to(self, device):
        self.base = self.base.to(device)
        self.instruction = self.instruction.to(device)
        self.joint = self.joint.to(device)
        return self

    def __del__(self):
        del self.base
        del self.instruction
        del self.joint
        th.cuda.empty_cache()


@th.no_grad()
def get_activations(batch, model, layer):
    nnmodel = NNsight(model)
    with nnmodel.trace(
        batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
    ):
        activations = nnmodel.model.layers[layer].output[0].save()
        nnmodel.model.layers[layer].output.stop()
    return activations.value


@th.no_grad()
def get_latents(batch, base_model, instruction_model, ae, layer):
    base_activations = get_activations(batch, base_model, layer)
    instruction_activations = get_activations(batch, instruction_model, layer)
    activations = th.stack([base_activations, instruction_activations], dim=-2).to(
        th.float32
    )
    batch_size = base_activations.shape[0]
    activations = activations.view(-1, activations.shape[-2], activations.shape[-1])
    latents_joint, latents_split = ae.encode(activations, return_no_sum=True)
    # rescale latents by decoder column norms
    rescaled_latents_split = latents_split * ae.decoder.weight.norm(dim=2).unsqueeze(0)
    rescaled_latents_joint = latents_joint * ae.decoder.weight.norm(dim=2).sum(
        dim=0, keepdim=True
    )

    base_latents = latents_split.view(
        batch_size, -1, latents_split.shape[-2], latents_split.shape[-1]
    )[..., 0, :]
    instruction_latents = latents_split.view(
        batch_size, -1, latents_split.shape[-2], latents_split.shape[-1]
    )[..., 1, :]

    base_latents_rescaled = rescaled_latents_split.view(
        batch_size,
        -1,
        rescaled_latents_split.shape[-2],
        rescaled_latents_split.shape[-1],
    )[..., 0, :]
    instruction_latents_rescaled = rescaled_latents_split.view(
        batch_size,
        -1,
        rescaled_latents_split.shape[-2],
        rescaled_latents_split.shape[-1],
    )[..., 1, :]

    return Latents(
        base=base_latents_rescaled,
        instruction=instruction_latents_rescaled,
        joint=rescaled_latents_joint,
    ), Latents(base=base_latents, instruction=instruction_latents, joint=latents_joint)


def filter_stack_latents(latents, attention_mask):
    # latents: (batch_size, seq_len, n_layers, dict_size)
    # attention_mask: (batch_size, seq_len)
    base_latents = latents.base.view(-1, latents.base.shape[-1])[
        attention_mask.view(-1).bool()
    ]
    instruction_latents = latents.instruction.view(-1, latents.instruction.shape[-1])[
        attention_mask.view(-1).bool()
    ]
    joint_latents = latents.joint.view(-1, latents.joint.shape[-1])[
        attention_mask.view(-1).bool()
    ]
    return Latents(base_latents, instruction_latents, joint_latents)


@dataclass
class LatentStatistic:
    avg_activation: th.Tensor
    non_zero_counts: th.Tensor
    max_activation: th.Tensor
    total_tokens: int
    is_normalized: bool = False

    def normalize(self):
        if self.is_normalized:
            return self
        self.avg_activation /= self.total_tokens
        self.is_normalized = True
        return self

    def combine(self, other):
        self.avg_activation += other.avg_activation
        self.non_zero_counts += other.non_zero_counts
        self.max_activation = th.maximum(self.max_activation, other.max_activation)
        self.total_tokens += other.total_tokens
        return self

    def to(self, device):
        self.avg_activation = self.avg_activation.to(device)
        self.non_zero_counts = self.non_zero_counts.to(device)
        self.max_activation = self.max_activation.to(device)
        return self


@dataclass
class LatentStatistics:
    base: LatentStatistic
    instruction: LatentStatistic
    joint: LatentStatistic

    abs_activation_diff: th.Tensor
    rel_activation_diff: th.Tensor
    either_non_zero_counts: th.Tensor
    is_normalized: bool = False

    def normalize(self):
        if self.is_normalized:
            return self
        self.base.normalize()
        self.instruction.normalize()
        self.joint.normalize()
        self.abs_activation_diff /= self.base.total_tokens
        self.rel_activation_diff /= self.either_non_zero_counts
        self.is_normalized = True
        return self

    def combine(self, other):
        self.base.combine(other.base)
        self.instruction.combine(other.instruction)
        self.joint.combine(other.joint)
        self.abs_activation_diff += other.abs_activation_diff
        self.rel_activation_diff += other.rel_activation_diff
        self.either_non_zero_counts += other.either_non_zero_counts
        return self

    def to(self, device):
        self.base = self.base.to(device)
        self.instruction = self.instruction.to(device)
        self.joint = self.joint.to(device)
        self.abs_activation_diff = self.abs_activation_diff.to(device)
        self.rel_activation_diff = self.rel_activation_diff.to(device)
        self.either_non_zero_counts = self.either_non_zero_counts.to(device)
        return self


@dataclass
class CombinedLatentStatistics:
    rescaled: LatentStatistics
    normal: LatentStatistics

    def normalize(self):
        self.rescaled.normalize()
        self.normal.normalize()
        return self

    def to(self, device):
        self.rescaled = self.rescaled.to(device)
        self.normal = self.normal.to(device)
        return self


def compute_statistics(latents, total_tokens, non_zero_threshold=1e-8):
    base_avg_activation = latents.base.sum(dim=0)
    instruction_avg_activation = latents.instruction.sum(dim=0)
    joint_avg_activation = latents.joint.sum(dim=0)

    base_max_activation = latents.base.max(dim=0).values
    instruction_max_activation = latents.instruction.max(dim=0).values
    joint_max_activation = latents.joint.max(dim=0).values

    base_non_zero_counts = (latents.base > non_zero_threshold).sum(dim=0)
    instruction_non_zero_counts = (latents.instruction > non_zero_threshold).sum(dim=0)
    joint_non_zero_counts = (latents.joint > non_zero_threshold).sum(dim=0)
    activation_diff = (latents.base - latents.instruction).sum(dim=0)
    rel_activation_diff = (
        (latents.base - latents.instruction)
        / (
            th.stack([latents.base, latents.instruction], dim=0).max(dim=0).values
            + 1e-8
        )
        + 1
    ) / 2
    # only consider latents that are non-zero in either base or instruction -> set others to 0
    rel_activation_diff = (
        rel_activation_diff
        * (
            (latents.base > non_zero_threshold)
            | (latents.instruction > non_zero_threshold)
        ).float()
    )
    rel_activation_diff = rel_activation_diff.sum(dim=0)
    either_non_zero_counts = (
        (latents.base > non_zero_threshold) | (latents.instruction > non_zero_threshold)
    ).sum(dim=0)

    base_statistic = LatentStatistic(
        base_avg_activation, base_non_zero_counts, base_max_activation, total_tokens
    )
    instruction_statistic = LatentStatistic(
        instruction_avg_activation,
        instruction_non_zero_counts,
        instruction_max_activation,
        total_tokens,
    )
    joint_statistic = LatentStatistic(
        joint_avg_activation, joint_non_zero_counts, joint_max_activation, total_tokens
    )
    return LatentStatistics(
        base=base_statistic,
        instruction=instruction_statistic,
        joint=joint_statistic,
        abs_activation_diff=activation_diff,
        rel_activation_diff=rel_activation_diff,
        either_non_zero_counts=either_non_zero_counts,
    )


def latent_statistics(
    dataset,
    tokenizer,
    base_model,
    instruction_model,
    ae,
    layer,
    batch_size=128,
    non_zero_threshold=1e-8,
    text_column="text",
):
    batch_idx = 0
    rescaled_stats = None
    normal_stats = None

    for batch in tqdm(DataLoader(dataset, batch_size=batch_size)):
        tokens = tokenizer(
            batch[text_column],
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,
        )
        batch_rescaled_latents, batch_normal_latents = get_latents(
            tokens, base_model, instruction_model, ae, layer
        )
        batch_rescaled_latents = filter_stack_latents(
            batch_rescaled_latents, tokens["attention_mask"]
        )
        batch_normal_latents = filter_stack_latents(
            batch_normal_latents, tokens["attention_mask"]
        )
        rescaled_stats_batch = compute_statistics(
            batch_rescaled_latents, tokens["attention_mask"].sum(), non_zero_threshold
        )
        normal_stats_batch = compute_statistics(
            batch_normal_latents, tokens["attention_mask"].sum(), non_zero_threshold
        )

        if batch_idx == 0:
            rescaled_stats = rescaled_stats_batch
            normal_stats = normal_stats_batch
        else:
            rescaled_stats.combine(rescaled_stats_batch)
            normal_stats.combine(normal_stats_batch)

        batch_idx += 1

    rescaled_stats.normalize()
    normal_stats.normalize()

    return CombinedLatentStatistics(rescaled_stats, normal_stats)


### Visualization ###


def add_relative_annotations(fig):
    y_offset = -0.08
    # add annotation
    fig.add_annotation(
        x=0.01,
        y=y_offset,
        text="(instr only)",
        showarrow=False,
        xref="paper",
        yref="paper",
    )

    # add annotation
    fig.add_annotation(
        x=0.99,
        y=y_offset,
        text="(base only)",
        showarrow=False,
        xref="paper",
        yref="paper",
    )
    fig.add_annotation(
        x=0.5,
        y=y_offset,
        text="(equal activation)",
        showarrow=False,
        xref="paper",
        yref="paper",
    )
    # x axis ticks
    fig.update_xaxes(tickmode="array", tickvals=[0.0, 0.5, 1.0])
    return fig


def filtered_stats(stats_fineweb, stats_lmsys, group_name, rescaled, indices_path):
    title_suffix = ""
    if group_name == "shared":
        title_suffix = "(Shared Latents)"
    elif group_name == "instruction":
        title_suffix = "(Instruction Only Latents)"
    elif group_name == "base":
        title_suffix = "(Base Only Latents)"
    # Extract latents for the last token position
    dead_indices = dead_latent_indices(
        combined_latent_statistics=[stats_fineweb, stats_lmsys], rescaled=rescaled
    )
    only_base_indices = th.load(
        f"{indices_path}/only_base_decoder_latent_indices.pt"
    ).cpu()
    only_it_indices = th.load(f"{indices_path}/only_it_decoder_latent_indices.pt").cpu()
    shared_indices = th.load(f"{indices_path}/shared_decoder_latent_indices.pt").cpu()

    stats_fineweb = stats_fineweb.rescaled if rescaled else stats_fineweb.normal
    stats_lmsys = stats_lmsys.rescaled if rescaled else stats_lmsys.normal

    if group_name == "shared":
        filter_indices = shared_indices
    elif group_name == "instruction":
        filter_indices = only_it_indices
    elif group_name == "base":
        filter_indices = only_base_indices
    else:
        group_name = "all"
        filter_indices = th.arange(stats_fineweb.rel_activation_diff.shape[0])

    return dead_indices, filter_indices, title_suffix


def plot_latent_diff(
    save_dir,
    stats_fineweb,
    stats_lmsys,
    group_name=None,
    rescaled=True,
    indices_path=None,
    save=True,
):
    fineweb_color = "#3f977a"
    lmsys_color = "#fd850d"
    # Create subplots
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=1, shared_yaxes=False)

    dead_indices, filter_indices, title_suffix = filtered_stats(
        stats_fineweb, stats_lmsys, group_name, rescaled, indices_path
    )

    stats_fineweb = stats_fineweb.rescaled if rescaled else stats_fineweb.normal
    stats_lmsys = stats_lmsys.rescaled if rescaled else stats_lmsys.normal

    fineweb_filtered_diff = remove_dead_and_filter(
        stats_fineweb.rel_activation_diff, dead_indices, filter_indices
    )
    lmsys_filtered_diff = remove_dead_and_filter(
        stats_lmsys.rel_activation_diff, dead_indices, filter_indices
    )

    # Full range histogram
    fineweb_hist = go.Histogram(
        x=fineweb_filtered_diff.cpu().numpy(),
        nbinsx=50,
        name="Fineweb",
        marker_color=fineweb_color,
    )
    fig.add_trace(fineweb_hist, row=1, col=1)
    lmsys_hist = go.Histogram(
        x=lmsys_filtered_diff.cpu().numpy(),
        nbinsx=50,
        name="LMSYS",
        marker_color=lmsys_color,
    )
    fig.add_trace(lmsys_hist, row=1, col=1)

    # Y axis 2 subplot
    fig.update_yaxes(row=1, col=1, type="log")
    fig.update_yaxes(row=1, col=1, title="Count")
    fig.update_xaxes(row=1, col=1, title="Latent Activation Difference")
    # Update layout
    fig.update_layout(
        title=f"<b>Average Relative Latent Activation Difference <br>{title_suffix}</b><br><sup>Number of Tokens: Fineweb {stats_fineweb.joint.total_tokens:.2e} - LMSYS {stats_lmsys.joint.total_tokens:.2e}</sup>",
    )

    # subtitle
    fig.update_layout(width=700, height=500)

    # legend position
    fig.update_layout(legend=dict(x=0.76, y=0.95))

    fig = add_relative_annotations(fig)

    # save
    if save:
        fig.write_image(
            f"{save_dir}/latent_diff_{group_name}{'_rescaled' if rescaled else ''}.png",
            scale=2,
        )
    return fig


def get_freq(stats, split="joint"):
    if split == "base":
        return stats.base.non_zero_counts / stats.base.total_tokens
    elif split == "instruction":
        return stats.instruction.non_zero_counts / stats.instruction.total_tokens
    elif split == "joint":
        return stats.joint.non_zero_counts / stats.joint.total_tokens
    else:
        raise ValueError(f"Invalid split: {split}")


def plot_latent_freq(
    save_dir,
    stats_fineweb,
    stats_lmsys,
    group_name=None,
    split="joint",
    rescaled=True,
    indices_path=None,
    topk=100,
    save=True,
):
    title_suffix = ""
    if group_name == "shared":
        title_suffix = " (Shared Latents)"
    elif group_name == "instruction":
        title_suffix = " (Instruction Only Latents)"
    elif group_name == "base":
        title_suffix = " (Base Only Latents)"

    fineweb_color = "#3f977a"
    lmsys_color = "#fd850d"
    # Create subplots
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False, horizontal_spacing=0.05)

    dead_indices, filter_indices, title_suffix = filtered_stats(
        stats_fineweb, stats_lmsys, group_name, rescaled, indices_path
    )
    stats_fineweb = stats_fineweb.rescaled if rescaled else stats_fineweb.normal
    stats_lmsys = stats_lmsys.rescaled if rescaled else stats_lmsys.normal

    fineweb_freq = get_freq(stats_fineweb, split=split)
    lmsys_freq = get_freq(stats_lmsys, split=split)

    fineweb_filtered_freq = remove_dead_and_filter(
        fineweb_freq, dead_indices, filter_indices
    )
    lmsys_filtered_freq = remove_dead_and_filter(
        lmsys_freq, dead_indices, filter_indices
    )

    sorted_indices = th.argsort(fineweb_filtered_freq, descending=True)[:topk]
    fig.add_trace(
        go.Bar(
            x=th.arange(topk),
            y=fineweb_filtered_freq[sorted_indices],
            name="Fineweb",
            marker_color=fineweb_color,
        ),
        row=1,
        col=1,
    )
    sorted_indices = th.argsort(lmsys_filtered_freq, descending=True)[:topk]
    fig.add_trace(
        go.Bar(
            x=th.arange(topk),
            y=lmsys_filtered_freq[sorted_indices],
            name="LMSYS",
            marker_color=lmsys_color,
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(title="Frequency")
    fig.update_xaxes(title="Topk Latents (Sorted)")

    fig.update_layout(
        title=f"<b>Latent Frequency {title_suffix}</b><br><sup>Number of Tokens: Fineweb {stats_fineweb.joint.total_tokens:.2e} - LMSYS {stats_lmsys.joint.total_tokens:.2e}</sup>"
    )

    fig.update_layout(width=900, height=500)
    fig.update_layout(legend=dict(x=0.76, y=0.95))
    if save:
        fig.write_image(
            f"{save_dir}/latent_freq_{group_name}_{split}{'_rescaled' if rescaled else ''}.png",
            scale=2,
        )
    return fig
