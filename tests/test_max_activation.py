import sys
from pathlib import Path
import pytest
import torch as th
from unittest.mock import patch

# Add the parent directory to the path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))

from scripts.compute_max_activation import (
    compute_max_activation,
    compute_max_activation_from_latent_cache,
)


class SimpleCrossCoder:
    """Simple CrossCoder for testing purposes."""

    def __init__(self, dict_size):
        self.dict_size = dict_size

    def get_activations(self, batch):
        """Return predictable activations for testing."""
        batch_size = len(batch)
        # Create a simple pattern where each feature has a known max value
        activations = th.zeros((batch_size, self.dict_size))
        for i in range(batch_size):
            # Each sample has increasing values for each feature
            activations[i] = th.tensor([i + j + 1.0 for j in range(self.dict_size)])
        return activations


class SimplePairedCache:
    """Simple activation cache for testing."""

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Just return the index - the actual value doesn't matter
        # as we're mocking the CrossCoder.get_activations method
        return th.tensor([idx])


class SimpleLatentCache:
    """Simple latent activation cache for testing."""

    def __init__(self, num_samples, dict_size, num_tokens=3, base_value=1.0):
        self.num_samples = num_samples
        self.dict_size = dict_size
        self.num_tokens = num_tokens
        self.base_value = base_value

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Return a sequence and its activations."""
        # Create a simple sequence
        sequence = th.tensor(list(range(1, self.num_tokens + 1)))

        # Create activations with a predictable pattern
        activations = th.zeros((self.num_tokens, self.dict_size))
        for i in range(self.num_tokens):  # tokens in sequence
            activations[i] = th.tensor(
                [idx + i + j + self.base_value for j in range(self.dict_size)]
            )

        return sequence, activations

    def to(self, device):
        # No-op for testing
        return self


@pytest.fixture(
    params=[
        # (dict_size, num_samples, num_tokens, base_value, batch_size)
        (50, 30, 30, 1.0, 20),  # Default case with bigger numbers
        (100, 50, 40, 0.5, 30),  # Larger dictionary, more samples, more tokens
        (30, 20, 20, 2.0, 10),  # Smaller dictionary, but still with bigger numbers
    ]
)
def activation_test_params(request):
    """Fixture to generate parameterized test data."""
    dict_size, num_samples, num_tokens, base_value, batch_size = request.param
    return {
        "dict_size": dict_size,
        "num_samples": num_samples,
        "num_tokens": num_tokens,
        "base_value": base_value,
        "batch_size": batch_size,
    }


def test_max_activation_computation(activation_test_params):
    """Test that max activation computation works correctly for both methods."""
    # Unpack test parameters
    dict_size = activation_test_params["dict_size"]
    num_samples = activation_test_params["num_samples"]
    num_tokens = activation_test_params["num_tokens"]
    base_value = activation_test_params["base_value"]
    batch_size = activation_test_params["batch_size"]

    device = "cpu"

    # Part 1: Test compute_max_activation_from_latent_cache
    print(
        f"\nTesting compute_max_activation_from_latent_cache with params: {activation_test_params}"
    )

    # Create test object with parameterized settings
    latent_cache = SimpleLatentCache(
        num_samples, dict_size, num_tokens=num_tokens, base_value=base_value
    )

    # Expected max values (based on our simple pattern)
    # For each feature j, the max will be (num_samples-1) + (num_tokens-1) + j + base_value
    expected_max = th.tensor(
        [num_samples - 1 + num_tokens - 1 + j + base_value for j in range(dict_size)],
        dtype=th.float32,
    )

    # Compute max activations
    max_acts_latent = compute_max_activation_from_latent_cache(latent_cache, device)

    # Verify results
    assert th.allclose(
        max_acts_latent, expected_max
    ), f"Max activation computation from latent cache failed: {max_acts_latent} vs {expected_max}"

    # Print the results for clarity
    print(f"Computed max activations from latent cache: {max_acts_latent}")
    print(f"Expected max activations: {expected_max}")

    # Part 2: Test compute_max_activation with equivalent data
    print("\nTesting compute_max_activation:")

    # Create a CrossCoder that will produce activations matching our latent cache
    class MatchingCrossCoder(SimpleCrossCoder):
        def get_activations(self, batch):
            """Return activations that match what's in the latent cache."""
            batch_size = len(batch)
            activations = th.zeros((batch_size, self.dict_size))
            for i in range(batch_size):
                # Get the sample index from the batch
                idx = batch[i].item()
                # For each sample, we need to produce an activation value that matches
                # the maximum value across all token positions in the latent cache
                # In SimpleLatentCache, the value at token position i is: idx + i + j + base_value
                # With num_tokens positions, the max is at i=num_tokens-1
                activations[i] = th.tensor(
                    [
                        idx + (num_tokens - 1) + j + base_value
                        for j in range(self.dict_size)
                    ]
                )
            return activations

    # Create test objects
    crosscoder = MatchingCrossCoder(dict_size)
    cache = SimplePairedCache(num_samples)

    # Mock the DataLoader to return our simple batches
    with patch("torch.utils.data.DataLoader", autospec=True) as mock_dataloader:
        # Configure the mock to return batches of indices
        mock_dataloader.return_value = [
            th.tensor(list(range(i, min(i + batch_size, num_samples))))
            for i in range(0, num_samples, batch_size)
        ]

        # Compute max activations
        max_acts_paired = compute_max_activation(
            crosscoder, cache, device, batch_size, num_workers=0
        )

        # Verify results
        assert th.allclose(
            max_acts_paired, expected_max
        ), f"Max activation computation failed: {max_acts_paired} vs {expected_max}"

        # Print the results for clarity
        print(f"Computed max activations from paired cache: {max_acts_paired}")
        print(f"Expected max activations: {expected_max}")
