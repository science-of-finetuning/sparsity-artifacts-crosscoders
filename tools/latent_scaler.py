import torch as th
import torch.nn as nn
from collections import namedtuple
from loguru import logger

from dictionary_learning.trainers.crosscoder import CrossCoderTrainer
from dictionary_learning.dictionary import CrossCoder

class FeatureScaler(th.nn.Module):
    def __init__(self, dict_size: int, fixed_mask: th.Tensor | None = None, zero_init: bool = False, use_elu: bool = True):
        super().__init__()
        self.dict_size = dict_size
        if fixed_mask is None:
            self.scaler = th.nn.Parameter(self.get_init_vector(dict_size, use_elu, zero_init))
            self.fixed_mask = None
        else:
            self.register_buffer('fixed_mask', fixed_mask)
            self.scaler = th.nn.Parameter(self.get_init_vector((~fixed_mask).sum(), use_elu, zero_init))
            self.fixed_mask = fixed_mask # only ~fixed_mask is trainable -> all other features are *fixed*
            th.save(self.fixed_mask, "fixed_mask_joint.pt")

        if use_elu:
            self.act_func = lambda x: th.nn.functional.elu(x) + 1
        else:
            self.act_func = th.nn.ReLU()

    def get_init_vector(self, size, use_elu: bool = False, init_zeros: bool = False, device: th.device = "cuda"):
        if use_elu:
            return th.zeros(size, device=device) if not init_zeros else th.ones(size, device=device) * -10
        else:
            return th.ones(size, device=device) if not init_zeros else th.zeros(size, device=device)

    def forward(self, features: th.Tensor):
        if self.fixed_mask is None:
            return features * self.act_func(self.scaler)
        else:
            # DEBUG print("fwd: prescaling min max", features[:, ~self.fixed_mask].max(), features[:, ~self.fixed_mask].min())
            # DEBUG print("index of max", features[:, ~self.fixed_mask].flatten().argmax())
            features[:, ~self.fixed_mask] *= self.act_func(self.scaler)
            # DEBUG print("fwd: postscaling min max", features[:, ~self.fixed_mask].max(), features[:, ~self.fixed_mask].min())
            return features
        
class IndividualFeatureScaler(th.nn.Module):
    def __init__(self, dict_size: int, feature_indices: th.Tensor | None = None, zero_init: bool = False, use_elu: bool = True):
        super().__init__()
        self.dict_size = dict_size
        self.scaler = th.nn.Parameter(self.get_init_vector(len(feature_indices), use_elu, zero_init))
        self.feature_indices = feature_indices

        if use_elu:
            self.act_func = lambda x:   
        else:
            self.act_func = th.nn.ReLU()

    def get_init_vector(self, size, use_elu: bool = False, init_zeros: bool = False, device: th.device = "cuda"):
        if use_elu:
            return th.zeros(size, device=device) if not init_zeros else th.ones(size, device=device) * -10
        else:
            return th.ones(size, device=device) if not init_zeros else th.zeros(size, device=device)

    def forward(self, features: th.Tensor):
        # features: (batch_size, num_features)
        batch_size, num_features = features.shape
        # DEBUG print("FWD: pre scaler min max", features.max(), features.min())
        features = features.unsqueeze(1).repeat(1, num_features, 1) # (batch_size, num_features, num_features)
        features[:, th.arange(num_features), th.arange(num_features)] *= self.act_func(self.scaler)
        # DEBUG print("FWD: post scaler min max", features.max(), features.min())
        # flatten & return 
        return features.reshape(batch_size*num_features, num_features)

class FeatureScalerTrainer(CrossCoderTrainer):
    def __init__(self, cross_coder: CrossCoder, target_decoder_layers: list[int] = None, feature_scaler: FeatureScaler | None = None, **kwargs):
        assert "pretrained_ae" not in kwargs, "pretrained_ae should not be set for FeatureScalerTrainer"
        assert "layer" not in kwargs, "layer should not be set for FeatureScalerTrainer"
        assert "lm_name" not in kwargs, "lm_name should not be set for FeatureScalerTrainer"
        assert "submodule_name" not in kwargs, "submodule_name should not be set for FeatureScalerTrainer"
        assert "resample_steps" not in kwargs, "resample_steps should not be set for FeatureScalerTrainer"
        self.compile = kwargs.pop("compile", False)
        super().__init__(**kwargs, pretrained_ae=cross_coder, layer=-1, lm_name="feature_scaler", compile=False)
        if feature_scaler is None:
            self.feature_scaler = FeatureScaler(cross_coder.dict_size)
        else:
            self.feature_scaler = feature_scaler

        if target_decoder_layers is None:
            self.target_decoder_layers = list(range(cross_coder.num_layers))
        else:
            self.target_decoder_layers = target_decoder_layers

        self.ae = CrossCoder(
            activation_dim=cross_coder.activation_dim,
            dict_size=cross_coder.dict_size,
            num_layers=cross_coder.num_layers,
            num_decoder_layers=len(target_decoder_layers),
            latent_processor=self.feature_scaler
        )

        # DEBUG print("init", self.ae.latent_processor)
        self.ae.encoder.weight = nn.Parameter(cross_coder.encoder.weight.data)
        self.ae.encoder.bias = nn.Parameter(cross_coder.encoder.bias.data)
        self.ae.decoder.weight = nn.Parameter(cross_coder.decoder.weight.data[target_decoder_layers, :, :])
        self.ae.decoder.bias = nn.Parameter(cross_coder.decoder.bias.data[target_decoder_layers, :])

        # disable gradients for ae
        for param in self.ae.parameters():
            param.requires_grad = False
        # reenable gradients for feature scaler
        for param in self.feature_scaler.parameters():
            param.requires_grad = True

        def warmup_fn(step):
            return min(step / self.warmup_steps, 1.0)


        
        if self.compile:
            self.ae = th.compile(self.ae)

        self.ae.to(self.device)
        self.optimizer = th.optim.Adam(self.feature_scaler.parameters(), lr=self.lr)
        self.scheduler = th.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_fn)


    def loss(self, x, logging=False, return_deads=False, **kwargs):
        # DEBUG print("activations", x.mean())

        x_hat, f = self.ae(x, output_features=True)

        l2_loss = th.linalg.norm(x[:, self.target_decoder_layers] - x_hat, dim=-1)
        print("l2_loss", l2_loss)
        print("l2_loss.mean()", l2_loss.shape, l2_loss.mean())
        l1_loss = f.norm(p=1, dim=-1).mean()
        deads = (f <= 1e-8).all(dim=0)
        if self.steps_since_active is not None:
            # update steps_since_active
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        loss = l2_loss + self.l1_penalty * l1_loss
        # DEBUG print("x_hat", x_hat)
        # DEBUG print("x", x)
        # DEBUG print("loss", loss)
        # DEBUG print("fsum", f.sum())
        # DEBUG print("l2_loss", l2_loss)
        # DEBUG print("l1_loss", l1_loss)
        # DEBUG print("decnorm", self.ae.decoder.weight.norm())
        # DEBUG print("encnorm", self.ae.encoder.weight.norm())
        # loss.backward()
        # # DEBUG print("scaler", self.feature_scaler.scaler.grad.norm())
        # exit()
        scalars = self.feature_scaler.act_func(self.feature_scaler.scaler)
        num_pos_scalars = (scalars > 1e-6).sum().item()
        num_scalars = scalars.numel()
        sparsity = num_pos_scalars / num_scalars
        min_scalars = scalars.min().item()
        max_scalars = scalars.max().item()
        mean_scalars = scalars.mean().item()

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : (x - x_hat).pow(2).sum(dim=-1).mean().item(),
                    'sparsity_loss' : l1_loss.item(),
                    'loss' : loss.item(),
                    'deads' : deads if return_deads else None,
                    'frac_active_scalars' : sparsity,
                    'scaler_min' : min_scalars,
                    'scaler_max' : max_scalars,
                    'scaler_mean' : mean_scalars
                }
            )

    @property
    def model(self):
        return self.feature_scaler

class IndividualFeatureScalerTrainer(CrossCoderTrainer):
    def __init__(self, cross_coder: CrossCoder, feature_indices: list[int], target_decoder_layer: int, source_decoder_weights: th.Tensor, zero_init: bool = False, dtype=th.float64, **kwargs):
        assert "pretrained_ae" not in kwargs, "pretrained_ae should not be set for FeatureScalerTrainer"
        assert "layer" not in kwargs, "layer should not be set for FeatureScalerTrainer"
        assert "lm_name" not in kwargs, "lm_name should not be set for FeatureScalerTrainer"
        assert "submodule_name" not in kwargs, "submodule_name should not be set for FeatureScalerTrainer"
        assert "resample_steps" not in kwargs, "resample_steps should not be set for FeatureScalerTrainer"
        self.compile = kwargs.pop("compile", False)
        super().__init__(**kwargs, pretrained_ae=cross_coder, layer=-1, lm_name="feature_scaler", compile=False)
        self._dtype = dtype
        
        self.target_decoder_layer = target_decoder_layer
        self.source_decoder_weights = source_decoder_weights.to(self.device)
        assert self.source_decoder_weights.shape == (len(feature_indices), cross_coder.activation_dim)

        self.ae = CrossCoder(
            activation_dim=cross_coder.activation_dim,
            dict_size=cross_coder.dict_size,
            num_layers=cross_coder.num_layers,
            num_decoder_layers=1,
        )
        # Modify the decoder weights to only decode the target decoder layer
        self.ae.encoder.weight = nn.Parameter(cross_coder.encoder.weight.data.to(dtype))
        self.ae.encoder.bias = nn.Parameter(cross_coder.encoder.bias.data.to(dtype))
        self.ae.decoder.weight = nn.Parameter(cross_coder.decoder.weight.data[[target_decoder_layer], :, :].to(dtype))
        self.ae.decoder.bias = nn.Parameter(cross_coder.decoder.bias.data[[target_decoder_layer], :].to(dtype))
        
        # Test truthfullness of the decoder weights
        th.set_printoptions(precision=10)
        if dtype == th.float64:
            logger.info("Testing correctness of copied decoder weights...")
            cross_coder.to(dtype)
            with th.no_grad():
                # If these fail, make sure that the matmul precision is set to highest
                assert self.ae.decoder.weight.shape == (1, self.ae.dict_size, self.ae.activation_dim)
                assert self.ae.decoder.bias.shape == (1, self.ae.activation_dim)
                assert th.all(self.ae.decoder.weight[0, :, :] == cross_coder.decoder.weight[target_decoder_layer, :, :])
                assert th.all(self.ae.decoder.bias[0, :] == cross_coder.decoder.bias[target_decoder_layer, :])
                test_input = th.randn(10, 2, cross_coder.activation_dim, device=self.device, dtype=dtype) # (batch_size, num_layers, activation_dim)
                
                #manual encoding
                f_ours = self.ae.encode(test_input).detach()
                f_target = cross_coder.encode(test_input).detach()
                assert f_ours.shape == f_target.shape
                assert th.allclose(f_ours, f_target)
                #manual decoding
                x_ours = self.ae.decode(f_ours)[:, 0]
                x_target = cross_coder.decode(f_target)[:, target_decoder_layer]
                assert x_ours.shape == x_target.shape
                assert th.allclose(x_ours, x_target)
                test_output_ours = self.ae(test_input)[:, 0]
                test_output_target = cross_coder(test_input)[:, target_decoder_layer]
                assert test_output_ours.shape == test_output_target.shape
                assert th.allclose(test_output_ours, test_output_target)
                logger.info("Correctness test passed")
        else:
            logger.warning("Skipping correctness tests for float32. Set dtype to float64 to run them.")

        # disable gradients for ae
        for param in self.ae.parameters():
            param.requires_grad = False
        
        self.feature_indices = th.tensor(feature_indices, device=self.device)
        self.feature_scaler = FeatureScaler(len(feature_indices), zero_init=zero_init)
        self.feature_scaler.to(self.device)
    
        self.feature_mask = th.zeros((cross_coder.dict_size), dtype=bool, device=self.device)
        self.feature_mask[feature_indices] = True
        self.feature_mask = self.feature_mask.to(self.device)
        
        if dtype == th.float64:
            logger.info("Testing correctness of the forward pass...")
            self.test_correctness()
            logger.info("Correctness test passed")

        def warmup_fn(step):
            return min(step / self.warmup_steps, 1.0)
        
        if self.compile:
            raise NotImplementedError("FeatureScalerTrainer does not support compilation")

        self.ae.to(self.device)
        self.optimizer = th.optim.Adam(self.feature_scaler.parameters(), lr=self.lr)
        self.scheduler = th.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_fn)

    # @th.no_grad()
    def test_correctness(self):
        feature_indices = self.feature_indices
        feature_mask = self.feature_mask
        source_decoder_weights = self.source_decoder_weights
        # Runs a test on the calculations. 
        # If we don't scale the target features and only look at a single feature, the following two computations should be equal
        activations = th.randn(100, 2, self.ae.activation_dim, dtype=self._dtype, device=self.device) # (batch_size, num_layers, activation_dim)
        
        toy_feature_mask = th.zeros((self.ae.dict_size), dtype=bool, device=self.device)
        toy_feature_mask[self.feature_indices[0]] = True
        self.feature_indices = [self.feature_indices[0]]
        self.feature_mask = toy_feature_mask
        self.source_decoder_weights = source_decoder_weights[[0]]

        x_hat_ours = self.forward(activations, scale_features=False)
        # 1. We decode the target features with the target decoder
        modified_crosscoder = CrossCoder(
            activation_dim=self.ae.activation_dim,
            dict_size=self.ae.dict_size,
            num_layers=self.ae.num_layers,
            num_decoder_layers=1,
        )
        modified_crosscoder.encoder.weight = nn.Parameter(self.ae.encoder.weight.data.to(self._dtype))
        modified_crosscoder.encoder.bias = nn.Parameter(self.ae.encoder.bias.data.to(self._dtype))
        modified_crosscoder.decoder.bias = nn.Parameter(self.ae.decoder.bias.data.to(self._dtype))
        modified_decoder_weights = th.clone(self.ae.decoder.weight.data.to(self._dtype))
        modified_decoder_weights[0, toy_feature_mask, :] = self.source_decoder_weights[[0]].to(self._dtype)
        modified_crosscoder.decoder.weight = nn.Parameter(modified_decoder_weights.to(self._dtype))

        x_hat_target = modified_crosscoder(activations)[:, 0]
        f = self.ae.encode(activations).detach()

        argmax = f[:, self.feature_indices].flatten().argmax()
        # DEBUG print("argmax", argmax)

        # DEBUG print("x_hat_ours", x_hat_ours[argmax])
        # DEBUG print("x_hat_ours", x_hat_ours.shape)
        # DEBUG print("x_hat_target", x_hat_target[argmax])
        # DEBUG print("x_hat_target", x_hat_target.shape)

        # DEBUG print("NORM DIFF", th.linalg.norm(x_hat_ours - x_hat_target, dim=-1).max())
        # DEBUG print("MAX DIFF", th.abs(x_hat_ours - x_hat_target).max())
        # DEBUG print("MEAN DIFF", th.abs(x_hat_ours - x_hat_target).mean())
        # DEBUG print("MEDIAN DIFF", th.median(th.abs(x_hat_ours - x_hat_target)))
        # DEBUG print("MIN DIFF", th.abs(x_hat_ours - x_hat_target).min())

        assert x_hat_ours.shape == x_hat_target.shape
        assert th.allclose(x_hat_ours, x_hat_target)
        self.feature_indices = feature_indices
        self.feature_mask = feature_mask
        

    def forward(self, activations, return_features=False, scale_features=True):
        # activations: (batch_size, num_layers, activation_dim)
        if self._dtype == th.float64:
            activations = activations.to(self._dtype)
        batch_size, num_layers, activation_dim = activations.shape
        f = self.ae.encode(activations).detach()
        # # DEBUG print("f", f[:, self.feature_indices], f.device)
        assert f.shape == (batch_size, self.ae.dict_size)
        
        target_reconstruction = self.ae.decode(f)[:, 0] # The crosscoder only has a single decoder layer -> we only need the first layer
        assert target_reconstruction.shape == (batch_size, activation_dim)

        f_target = f[:, self.feature_mask]
        assert f_target.shape == (batch_size, len(self.feature_indices))
        # DEBUG print("f_target", f_target.device)

        # Compute the target decoding only for the target features
        # First let us get only the source decoder for the target features
        target_decoder_target = self.ae.decoder.weight[0, self.feature_mask, :]
        assert target_decoder_target.shape == (len(self.feature_indices), activation_dim)
        # Now we repeat the target features for each batch element
        target_decoder_target = target_decoder_target.repeat(batch_size, 1)
        assert target_decoder_target.shape == (batch_size*len(self.feature_indices), activation_dim)
         # We now stack the target features
        f_target_reshaped = f_target.reshape(-1, 1)
        assert f_target_reshaped.shape == (batch_size*len(self.feature_indices), 1)
        # f_target is now a column vector 
        # [[batch_1_target_1], [batch_1_target_2], ... [batch_2_target_1], [batch_2_target_2], ...]
        # And then scale each decoder with the target features for all batches
        target_reconstruction_target = target_decoder_target * f_target_reshaped
        assert target_reconstruction_target.shape == (batch_size*len(self.feature_indices), activation_dim)

        # Now we repeat this but with the source decoder weights (len(feature_indices), activation_dim)
        # But this time we scale the f_target
        source_decoder_target = self.source_decoder_weights
        assert source_decoder_target.shape == (len(self.feature_indices), activation_dim)    
        # We repeat this for each batch element
        source_decoder_target = source_decoder_target.repeat(batch_size, 1)
        assert source_decoder_target.shape == (batch_size*len(self.feature_indices), activation_dim)
        if scale_features:
            f_target = self.feature_scaler(f_target)
            assert f_target.shape == (batch_size, len(self.feature_indices))
        # DEBUG print("f_target", f_target.device)
        
        f_target_reshaped = f_target.reshape(-1, 1)
        assert f_target_reshaped.shape == (batch_size*len(self.feature_indices), 1)
        # DEBUG print("f_target_reshaped", f_target_reshaped.device)
        source_reconstruction_target = source_decoder_target * f_target_reshaped
        assert source_reconstruction_target.shape == (batch_size*len(self.feature_indices), activation_dim)

        # We remove the target features and then add back in the scaled source features
        # But first we need to interleave the target_reconstruction
        target_reconstruction_interleaved = target_reconstruction.repeat_interleave(len(self.feature_indices), dim=0)
        assert target_reconstruction_interleaved.shape == (batch_size*len(self.feature_indices), activation_dim)
        x_hat = target_reconstruction_interleaved - target_reconstruction_target + source_reconstruction_target
        assert x_hat.shape == (batch_size*len(self.feature_indices), activation_dim)

        if return_features:
            f[:, self.feature_mask] = f_target
            return x_hat, f
        else:
            return x_hat

    def update(self, step, activations):
        activations = activations.to(self.device)
        x_hat = self.forward(activations)
        loss = self.loss(activations, x_hat)
        loss.backward()
        # DEBUG print("scaler", self.feature_scaler.scaler.grad.norm())

        # DEBUG print("scaler", self.feature_scaler.scaler)
        self.optimizer.step()
        self.scheduler.step()
        # DEBUG print("scaler", self.feature_scaler.scaler)

    def loss(self, x, x_hat=None, logging=False, **kwargs):
        if x_hat is None:
            x_hat, f = self.forward(x, return_features=True)
        # DEBUG print("x_hat", x_hat)
        # DEBUG print("x", x)
        batch_size, num_layers, activation_dim = x.shape
        x = x[:, self.target_decoder_layer].repeat_interleave(len(self.feature_indices), dim=0)

        assert x.shape == x_hat.shape
        l2_loss = th.linalg.norm(x - x_hat, dim=-1).sum() / batch_size
        # DEBUG print("l2_loss", l2_loss)
        # DEBUG print("decnorm", self.ae.decoder.weight.norm())
        # DEBUG print("encnorm", self.ae.encoder.weight.norm())

        # l2_loss.backward()
        # # DEBUG print("scaler", self.feature_scaler.scaler.grad.norm())
        # exit()
        scalars = self.feature_scaler.act_func(self.feature_scaler.scaler)
        num_pos_scalars = (scalars > 1e-6).sum().item()
        num_scalars = scalars.numel()
        sparsity = num_pos_scalars / num_scalars
        min_scalars = scalars.min().item()
        max_scalars = scalars.max().item()
        mean_scalars = scalars.mean().item()

        if not logging:
            return l2_loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f, 
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : (x - x_hat).pow(2).sum(dim=-1).mean().item(),
                    'loss' : l2_loss.item(),
                    'frac_active_scalars' : sparsity,
                    'scaler_min' : min_scalars,
                    'scaler_max' : max_scalars,
                    'scaler_mean' : mean_scalars,
                    # needed to use the train_SAE script from dictionary_learning
                    "deads" : None,
                    # needed to use the train_SAE script from dictionary_learning
                    "frac_deads" : th.zeros(x.shape[0]) 
                }
            )

    @property
    def model(self):
        return self.feature_scaler