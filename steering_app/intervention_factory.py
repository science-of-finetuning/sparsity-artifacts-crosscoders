import torch as th
from typing import Any
import sys
from pathlib import Path

# Add the parent directory to the path to import tools
sys.path.append(str(Path(__file__).parent.parent))
from tools.cc_utils import load_dictionary_model, load_latent_df
from tools.halfway_interventions import (
    HalfStepPreprocessFn,
    IdentityPreprocessFn,
    SwitchPreprocessFn,
    TestMaskPreprocessFn,
    PatchProjectionFromDiff,
    SteeringVector,
    CrossCoderReconstruction,
    CrossCoderSteeringLatent,
    CrossCoderAdditiveSteering,
    CrossCoderOutProjection,
    SAEAdditiveSteering,
    SAEDifferenceReconstruction,
    BasePatchIntervention,
    PatchCtrl,
    PatchKFirstPredictions,
    PatchKFirstAndCtrl,
    SteerWithCrosscoderLatent,
)


class DummyCrossCoder:
    """Dummy CrossCoder class for demonstration purposes"""

    def __init__(self):
        self.encoder = th.nn.Linear(768, 128)
        self.decoder = th.nn.Linear(128, 768)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class DummySAE:
    """Dummy SAE class for demonstration purposes"""

    def __init__(self):
        self.encoder = th.nn.Linear(768, 128)
        self.decoder = th.nn.Linear(128, 768)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def create_intervention(
    intervention_type: str,
    params: dict[str, Any],
    model_dict: dict[str, Any],
    cc_device: str,
) -> HalfStepPreprocessFn:
    """
    Create an intervention object based on the type and parameters

    Args:
        intervention_type: The class name of the intervention
        params: Dictionary of parameters for the intervention

    Returns:
        An instance of the specified intervention
    """
    # Create dummy objects for demonstration

    # Replace placeholder values with actual objects
    print(f"Creating intervention with params: {params}")
    params = params.copy()
    if "crosscoder" in params:
        print(f"Creating CrossCoder intervention with name {params['crosscoder']}")
        crosscoder_name = params.pop("crosscoder")
        if crosscoder_name not in model_dict:
            model_dict[crosscoder_name] = load_dictionary_model(crosscoder_name).to(
                cc_device
            )
        crosscoder = model_dict[crosscoder_name]

    if "sae" in params:
        print(f"Creating SAE intervention with name {params['sae']}")
        sae_name = params.pop("sae")
        if sae_name not in model_dict:
            model_dict[sae_name] = load_dictionary_model(sae_name, is_sae=True).to(
                cc_device
            )
        sae = model_dict[sae_name]

    # Create the intervention based on type
    if intervention_type == "IdentityPreprocessFn":
        return IdentityPreprocessFn(**params)

    elif intervention_type == "SwitchPreprocessFn":
        return SwitchPreprocessFn(**params)

    elif intervention_type == "TestMaskPreprocessFn":
        return TestMaskPreprocessFn(**params)

    elif intervention_type == "PatchProjectionFromDiff":
        return PatchProjectionFromDiff(**params)

    elif intervention_type == "SteeringVector":
        return SteeringVector(**params)

    elif intervention_type == "CrossCoderReconstruction":
        return CrossCoderReconstruction(**params)

    elif intervention_type == "CrossCoderSteeringLatent":
        return CrossCoderSteeringLatent(**params)

    elif intervention_type == "CrossCoderAdditiveSteering":
        return CrossCoderAdditiveSteering(**params)

    elif intervention_type == "CrossCoderOutProjection":
        return CrossCoderOutProjection(**params)

    elif intervention_type == "SAEAdditiveSteering":
        return SAEAdditiveSteering(sae=sae, **params)

    elif intervention_type == "SAEDifferenceReconstruction":
        return SAEDifferenceReconstruction(sae=sae, **params)

    elif intervention_type == "PatchCtrl":
        return PatchCtrl(**params)

    elif intervention_type == "PatchKFirstPredictions":
        return PatchKFirstPredictions(**params)

    elif intervention_type == "PatchKFirstAndCtrl":
        return PatchKFirstAndCtrl(**params)

    elif intervention_type == "SteerWithCrosscoderLatent":
        latent_df = load_latent_df(crosscoder_name)
        return SteerWithCrosscoderLatent(
            crosscoder=crosscoder, latent_df=latent_df, **params
        )

    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
