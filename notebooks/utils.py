import torch as th
from tqdm.auto import tqdm
from torch.nn.functional import cosine_similarity


def compute_chunked_cosine_similarity(weights1, weights2, chunk_size):
    # Calculate chunk size
    num_chunks = weights1.shape[0] // chunk_size

    # Create list to store chunk matrices
    cosim_matrices = []

    # Process each chunk
    for i in tqdm(range(num_chunks)):
        # th.cuda.empty_cache()
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else weights1.shape[0]
        chunk = weights1[start_idx:end_idx]

        # Compute cosine similarity for this chunk
        # Use modulo to cycle through available GPUs
        gpu_idx = i % th.cuda.device_count()
        device = f"cuda:{gpu_idx}"
        if gpu_idx == 0:
            # sync
            for id in range(th.cuda.device_count()):
                th.cuda.synchronize(f"cuda:{id}")
            th.cpu.synchronize()
        cosim_matrix_chunk = cosine_similarity(
            chunk.unsqueeze(1).to(device, non_blocking=True),
            weights2.unsqueeze(0).to(device, non_blocking=True),
            dim=2,
        ).to("cpu", non_blocking=True)
        cosim_matrices.append(cosim_matrix_chunk)

    # Combine all chunks and move to CPU
    cosim_matrix = th.cat(cosim_matrices, dim=0)

    return cosim_matrix


def create_control_mask(
    input: dict,
    eot_token: None | int = None,
) -> th.Tensor:
    # Handle both batched and unbatched inputs
    tokens = input["input_ids"]
    assistant_mask = th.tensor(input["assistant_masks"]).to(tokens.device)
    batch_size, seq_len = assistant_mask.shape

    # Create output tensor, starting with zeros
    control_mask = th.zeros_like(assistant_mask, dtype=th.bool)
    control_mask[:, 1:4] = 1

    # Process each batch item separately
    for b in range(batch_size):
        # Find transitions (0->1 and 1->0)
        transitions = assistant_mask[b, 1:] != assistant_mask[b, :-1]
        transition_indices = th.where(transitions)[0]

        # Group transitions into pairs (start, end)
        for i in range(0, len(transition_indices), 2):

            start_idx = transition_indices[i]
            if i + 1 >= len(transition_indices):
                end_idx = seq_len - 1
            else:
                end_idx = transition_indices[i + 1]

            # Keep the last 1 of the sequence
            if tokens is not None and eot_token is not None:
                control_mask[b, end_idx] = tokens[b, end_idx] == eot_token
            else:
                control_mask[b, end_idx] = 1

            # Set next 4 positions to 1 (if within bounds)
            next_four = min(end_idx + 4, seq_len)
            control_mask[b, end_idx + 1 : next_four + 1] = 1

            # Set preceding 5 positions to 1 (if within bounds)
            prev_five = max(start_idx - 4, 0)
            control_mask[b, prev_five : start_idx + 1] = 1

    return control_mask & input["attention_mask"]


from tiny_dashboard.html_utils import create_token_html, create_example_html, create_base_html
from tiny_dashboard.utils import sanitize_tokens, sanitize_token


def activation_visualization(
    tokens: list[str],
    activations: th.Tensor,
    tokenizer,
    highlight_idx: int | None = None,
    title: str = "",
) -> str:
    """Create HTML with highlighted tokens based on activation values"""
    html_parts = []
    # all_feature_indicies = list(range(activations.shape[0]))
    # Find highlight feature index in the activation tensor
    if highlight_idx is None:
        if activations.dim() == 2:
            raise ValueError(
                "Activations must be 1D unless a highlight feature is specified"
            )
        highlight_acts = activations
        activations = activations.unsqueeze(0)
        other_features = [0]
    else:
        highlight_acts = activations[highlight_idx]
        other_features = [i for i in range(activations.shape[0]) if i != highlight_idx]
    # Normalize activations for color intensity (only for highlight feature)
    max_highlight = highlight_acts.max()
    norm_acts = highlight_acts / (max_highlight + 1e-6)

    # Create HTML spans with activation values
    sanitized_tokens = sanitize_tokens(tokens, non_breaking_space=False)
    for i, (san_token, token) in enumerate(zip(sanitized_tokens, tokens)):

        color = f"rgba(255, 0, 0, {norm_acts[i].item():.3f})"

        # Create tooltip content only for requested features
        tok_id = tokenizer.convert_tokens_to_ids(token)
        tooltip_token = sanitize_token(
            token, keep_newline=False, non_breaking_space=False
        )
        tooltip_lines = [f"Token {tok_id}: '{tooltip_token}'"]
        for feat in other_features:
            act_value = activations[feat, i].item()
            tooltip_lines.append(f"Feature {feat}: {act_value:.3f}")

        tooltip_content = "\n".join(tooltip_lines)
        html_parts.append(create_token_html(san_token, color, tooltip_content))

    html = "".join(html_parts)
    html = create_example_html(max_highlight.item(), html, static=True)
    return create_base_html(title, html)
