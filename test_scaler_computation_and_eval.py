import torch as th
from dictionary_learning import CrossCoder
from tools.latent_scaler.closed_form import (
    remove_latents,
    closed_form_scalars,
    run_tests,
)
from scripts.verify_scalers import compute_stats

def main():
    th.set_grad_enabled(False)

    # Initialize parameters
    dict_size = 256
    dim_model = 64
    target_idx = 10
    BETA = 10.0

    # Initialize CrossCoder
    coder = CrossCoder(activation_dim=dim_model, dict_size=dict_size, num_layers=2)

    # Average encoder norm
    coder.encoder.weight *= 100
    encoder_norm = coder.encoder.weight.norm(dim=1).mean()
    print(f"Average encoder norm: {encoder_norm}")

    # Check decoder weight norm shape
    decoder_norm_shape = coder.decoder.weight.norm(dim=2).shape
    print(f"Decoder weight norm shape: {decoder_norm_shape}")

    # Average decoder norm
    coder.decoder.weight *= 100
    coder.decoder.bias *= 0
    decoder_norm = coder.decoder.weight.norm(dim=2).mean()
    print(f"Average decoder norm: {decoder_norm}")

    target_vector = coder.decoder.weight[1, target_idx]
    print(f"Target vector shape: {target_vector.shape}")
    normalized_target_vector = target_vector / target_vector.norm()
    projection_matrix = th.outer(normalized_target_vector.double(), normalized_target_vector.double())

    # Generate toy input and activations
    toy_input = th.randn(1000, 2, dim_model, dtype=th.float64)
    # Remove target_vector from toy_input base (is added back in later with beta)
    toy_input[:, 0, :] -= toy_input[:, 0, :] @ projection_matrix.T
    toy_input = toy_input.float()
    toy_activations = coder.encode(toy_input)
    print(f"Toy activations shape: {toy_activations.shape}")

    max_activation, active_count = toy_activations[:, target_idx].max(), (toy_activations[:, target_idx] > 0).sum()
    print(f"Max activation: {max_activation}, Active count: {active_count}")

    # Replace decoder weights for target_idx
    coder.decoder.weight[0, target_idx] = target_vector

    # Make other decoder vectors orthogonal to target_vector
    for layer in range(coder.num_layers):
        for i in range(dict_size):
            if i != target_idx:
                decoder_vector = coder.decoder.weight[layer, i].double()
                coder.decoder.weight[layer, i] = (decoder_vector - projection_matrix @ decoder_vector).float()
                assert (coder.decoder.weight[layer, i] @ target_vector) < 1e-5

    print("All non-target decoder vectors are orthogonal to the target vector.")

    # Define functions
    def load_base_error(batch, crosscoder: CrossCoder, latent_activations: th.Tensor, latent_indices: th.Tensor, **kwargs):
        reconstruction = crosscoder.decode(latent_activations)
        addon = latent_activations[:, target_idx].unsqueeze(1).cuda() * BETA * latent_vectors.repeat(latent_activations.shape[0], 1)
        return batch[:, 0, :] - remove_latents(
            reconstruction[:, 0, :],
            latent_activations[:, latent_indices],
            base_decoder[latent_indices],
        ) + addon

    def load_base_activation(batch, crosscoder: CrossCoder, latent_activations: th.Tensor, latent_indices: th.Tensor, latent_vectors: th.Tensor, **kwargs):
        addon = latent_activations[:, target_idx].unsqueeze(1).cuda() * BETA * latent_vectors.repeat(latent_activations.shape[0], 1)
        return batch[:, 0, :] + addon

    def load_base_reconstruction(batch, crosscoder: CrossCoder, latent_activations: th.Tensor, latent_indices: th.Tensor, latent_vectors: th.Tensor, **kwargs):
        reconstruction = crosscoder.decode(latent_activations)
        return remove_latents(
            reconstruction[:, 0, :], latent_activations[:, latent_indices], latent_vectors
        )

    # Setup decoder and latent vectors
    base_decoder = coder.decoder.weight[0, :, :].clone().cuda()
    chat_only_indices = th.tensor([target_idx]).cuda()
    latent_vectors = coder.decoder.weight[1, :, :].clone().cuda()
    latent_vectors = latent_vectors[chat_only_indices].cuda()

    # Create DataLoader
    dataloader = th.utils.data.DataLoader(toy_input, batch_size=10, shuffle=False)

    # Move coder to GPU
    coder.to("cuda")
    print(coder)

    # Compute closed-form scalars
    betas, count_active = closed_form_scalars(
        latent_vectors,
        chat_only_indices,
        dataloader,
        coder,
        load_base_error,
        device="cuda",
        dtype=th.float32,
        latent_activation_postprocessing_fn=None,
    )
    print(f"Betas: {betas}")

    assert th.isclose(betas[0], th.tensor(BETA), atol=1e-3), f"Computed beta {betas[0]} is not close to {BETA}"

    # Replace computed betas with BETA
    betas = th.tensor([BETA]).cuda()

    # Compute stats
    stats = compute_stats(
        betas=betas,
        latent_vectors=latent_vectors,
        latent_indices=chat_only_indices,
        dataloader=dataloader,
        crosscoder=coder,
        activation_postprocessing_fn=load_base_reconstruction,
        target_fn=load_base_activation,
        device="cuda",
        dtype=th.float32,
    )
    print(f"Stats: {stats}")

    # Further computations
    target_activations = load_base_activation(toy_input.cuda(), coder, toy_activations, chat_only_indices, latent_vectors)
    print(f"Target activations shape: {target_activations.shape}")

    latent_activations = coder.encode(toy_input.cuda())
    print(f"Latent activations shape: {latent_activations.shape}")

    latent_activations_for_before = latent_activations.clone()
    latent_activations_for_before[:, chat_only_indices] = 0  # Remove the base decoder vector
    reconstruction = coder.decode(latent_activations_for_before)
    print(f"Reconstruction shape: {reconstruction.shape}")

    mse_before = (target_activations - reconstruction[:, 0, :]).pow(2).mean()
    print(f"MSE before: {mse_before}")

    addon = latent_activations[:, target_idx].unsqueeze(1).cuda() * 10 * latent_vectors.repeat(latent_activations.shape[0], 1)
    reconstruction_after = reconstruction[:, 0, :] + addon
    mse_after = (target_activations - reconstruction_after).pow(2).mean()
    print(f"MSE after: {mse_after}")

    assert th.isclose(th.tensor(mse_before).float(), th.tensor(stats["mse_before"][0]).float(), atol=1e-3), f"MSE before {mse_before} is not close to MSE after {mse_after}"
    assert th.isclose(th.tensor(mse_after).float(), th.tensor(stats["mse"][0]).float(), atol=1e-3), f"MSE after {mse_after} is not close to MSE after {mse_after}"

    max_activations = latent_activations.max(0).values.cuda()
    print(f"Max activations shape: {max_activations.shape}")

    # Recompute stats with max_activations
    stats = compute_stats(
        max_activations=max_activations,
        betas=betas,
        latent_vectors=latent_vectors,
        latent_indices=chat_only_indices,
        dataloader=dataloader,
        crosscoder=coder,
        activation_postprocessing_fn=load_base_reconstruction,
        target_fn=load_base_activation,
        device="cuda",
        dtype=th.float32,
    )
    print(f"Updated Stats: {stats}")

    bucket_edges = th.tensor(stats["bucket_edges"]).to("cuda")
    print(f"Bucket edges: {bucket_edges}")

    # Apply masking and compute MSE for specific buckets
    for i in range(bucket_edges.shape[1] - 1):
        print(f"Bucket {i}")
        mask_activations = (latent_activations[:, target_idx] > bucket_edges[:, i]) & (latent_activations[:, target_idx] <= bucket_edges[:, i+1])
        print(f"Masked activations count and shape: {mask_activations.sum()}, {mask_activations.shape}")

        addon = latent_activations[mask_activations, target_idx].unsqueeze(1).cuda() * 10 * latent_vectors.repeat(mask_activations.sum(), 1)

        reconstruction_after = reconstruction[mask_activations, 0, :] + addon
        mse_before = (target_activations[mask_activations] - reconstruction[mask_activations, 0, :]).pow(2).mean()
        mse_after = (target_activations[mask_activations] - reconstruction_after).pow(2).mean()
        print(f"MSE after: {mse_after}")
        assert th.isclose(th.tensor(mse_after).float(), th.tensor(stats["mse_buckets"][i][0]).float(), atol=1e-3), f"MSE after {mse_after} is not close to {stats['mse_buckets'][i][0]}"
        assert th.isclose(th.tensor(mse_before).float(), th.tensor(stats["mse_before_buckets"][i][0]).float(), atol=1e-3), f"MSE before {mse_before} is not close to {stats['mse_before_buckets'][i][0]}"

if __name__ == "__main__":
    main() 