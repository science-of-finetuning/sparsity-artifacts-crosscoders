import torch as th
from typing import Dict, Any, List, Optional, Literal
import sys
from pathlib import Path

# Add the parent directory to the path to import tools
sys.path.append(str(Path(__file__).parent.parent))

# Import the halfway interventions module
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
    BasePatchIntervention,
    PatchCtrl,
    PatchKFirstPredictions,
    PatchKFirstAndCtrl,
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

def create_intervention(intervention_type: str, params: Dict[str, Any]) -> HalfStepPreprocessFn:
    """
    Create an intervention object based on the type and parameters
    
    Args:
        intervention_type: The class name of the intervention
        params: Dictionary of parameters for the intervention
        
    Returns:
        An instance of the specified intervention
    """
    # Create dummy objects for demonstration
    dummy_crosscoder = DummyCrossCoder()
    dummy_sae = DummySAE()
    
    # Replace placeholder values with actual objects
    if "crosscoder" in params and params["crosscoder"] == "dummy_crosscoder":
        params["crosscoder"] = dummy_crosscoder
    
    if "sae" in params and params["sae"] == "dummy_sae":
        params["sae"] = dummy_sae
    
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
        return SAEAdditiveSteering(**params)
    
    elif intervention_type == "PatchCtrl":
        return PatchCtrl(**params)
    
    elif intervention_type == "PatchKFirstPredictions":
        return PatchKFirstPredictions(**params)
    
    elif intervention_type == "PatchKFirstAndCtrl":
        return PatchKFirstAndCtrl(**params)
    
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")

def dummy_text_generation(intervention: HalfStepPreprocessFn, text: str, title: str = "Response") -> str:
    """
    Dummy function to simulate text generation with an intervention
    
    Args:
        intervention: The intervention object
        text: Input text
        title: The title to use for the response
        
    Returns:
        Generated text (dummy implementation) with markdown formatting
    """
    # In a real implementation, this would use the intervention to generate text
    intervention_name = intervention.__class__.__name__
    
    # Format the parameters in a cleaner way
    params_str = ""
    for key, value in intervention.__dict__.items():
        if key.startswith("_"):
            continue
        
        if isinstance(value, th.Tensor):
            params_str += f"\n- **{key}**: Tensor(shape={list(value.shape)})"
        else:
            params_str += f"\n- **{key}**: `{value}`"
    
    # Generate a dummy response based on the input text
    if not text or text == "Enter your text here...":
        response = "Please provide some input text to generate a response."
    else:
        # Create a simulated response that varies slightly based on the intervention type
        if intervention_name == "IdentityPreprocessFn":
            response = f"""## {title}

Your input: "{text}"

This response demonstrates the standard behavior of the model without any modifications to its activations.

Some example markdown formatting:
- Bullet point 1
- Bullet point 2

```python
# Some example code
def hello_world():
    print("Hello, world!")
```
"""
        elif "Switch" in intervention_name:
            response = f"""## {title}

Your input: "{text}"

This response demonstrates the effect of **switching** between base and chat models during generation.

The switch intervention can help understand how different models process the same input differently.

```
Base model → Chat model
or
Chat model → Base model
```
"""
        elif "Patch" in intervention_name:
            response = f"""## {title}

Your input: "{text}"

This response shows how **patching specific activations** affects the generation process.

Patching interventions can reveal which parts of the model are responsible for specific behaviors.

| Patch Type | Effect |
|------------|--------|
| Control tokens | Changes how instructions are processed |
| First K tokens | Affects initial generation trajectory |
| Combined | Most comprehensive intervention |
"""
        elif "Steering" in intervention_name:
            response = f"""## {title}

Your input: "{text}"

This response demonstrates how **steering vectors** influence the output of the model.

Steering can push the model's behavior in specific directions:
1. Enhance certain qualities
2. Suppress unwanted behaviors
3. Guide the model toward particular topics or styles

*The exact effect depends on the steering vector used.*
"""
        elif "CrossCoder" in intervention_name:
            response = f"""## {title}

Your input: "{text}"

This response shows the effect of **CrossCoder-based interventions** on model behavior.

CrossCoders allow for sophisticated manipulations of model activations by:
- Encoding activations into a latent space
- Modifying specific latent dimensions
- Decoding back to the original activation space

> This can lead to more targeted and interpretable interventions.
"""
        else:
            response = f"""## {title}

Your input: "{text}"

This is a simulated response using a specialized intervention technique.

The intervention modifies how the model processes information internally,
potentially changing its behavior in subtle or significant ways.
"""
    
    # Add intervention details at the bottom with markdown formatting
    intervention_info = f"""
---

### Intervention Details
**Type**: `{intervention_name}`

**Parameters**: {params_str}
"""
    
    return response + intervention_info 