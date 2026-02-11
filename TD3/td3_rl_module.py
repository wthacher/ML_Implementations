from typing import Any, Dict
import torch

from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import (
    DefaultSACTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.sac.sac_learner import (
    ACTION_DIST_INPUTS_NEXT,
    ACTION_LOG_PROBS,
    ACTION_LOG_PROBS_NEXT,
    ACTION_PROBS,
    ACTION_PROBS_NEXT,
    QF_PREDS,
    QF_TARGET_NEXT,
    QF_TWIN_PREDS,
)
from ray.rllib.core.models.base import ENCODER_OUT
from td3_catalog import TD3Catalog

class TD3TorchRLModule(DefaultSACTorchRLModule):
    """
    TD3 RLModule - Deterministic version of SAC.
    
    Changes from SAC:
    1. Use mean of distribution instead of sampling (rsample → mean)
    2. Remove log_prob computations (not needed without entropy)
    """
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = TD3Catalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)
    
    @override(DefaultSACTorchRLModule)
    def _forward_train_continuous(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        TD3 training forward pass.
        
        Key difference from SAC: Use deterministic actions (mean) instead of sampling.
        """
        output = {}
        
        batch_curr = {Columns.OBS: batch[Columns.OBS]}
        batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}
        
        # Encoder forward passes
        pi_encoder_outs = self.pi_encoder(batch_curr)
        pi_encoder_next_outs = self.pi_encoder(batch_next)
        
        # Q-network forward passes (same as SAC)
        batch_curr.update({Columns.ACTIONS: batch[Columns.ACTIONS]})
        output[QF_PREDS] = self._qf_forward_train_helper(
            batch_curr, self.qf_encoder, self.qf
        )
        if self.twin_q:
            output[QF_TWIN_PREDS] = self._qf_forward_train_helper(
                batch_curr, self.qf_twin_encoder, self.qf_twin
            )
        
        # Policy head outputs
        action_logits = self.pi(pi_encoder_outs[ENCODER_OUT])
        action_logits_next = self.pi(pi_encoder_next_outs[ENCODER_OUT])
        output[Columns.ACTION_DIST_INPUTS] = action_logits
        output[ACTION_DIST_INPUTS_NEXT] = action_logits_next
        
        # Get action distribution class
        # For TD3, we use the deterministic distribution
        action_dist_class = self.get_train_action_dist_cls()
        
        action_dist_curr = action_dist_class(action_logits)
        action_dist_next = action_dist_class(action_logits_next)
        
        actions_curr = action_dist_curr.sample()
        actions_next = action_dist_next.sample()
        
        # For TD3, we don't need log probabilities (no entropy term)
        # But we'll keep them as zeros for compatibility with the learner
        output["logp_resampled"] = torch.zeros(
            actions_curr.shape[0], device=actions_curr.device
        )
        output["logp_next_resampled"] = torch.zeros(
            actions_next.shape[0], device=actions_next.device
        )
        
        # Compute Q-values for current policy in current state
        q_batch_curr = {
            Columns.OBS: batch[Columns.OBS],
            Columns.ACTIONS: actions_curr,
        }
        
        # Straight-through gradient for Q-network (same as SAC)
        all_params = list(self.qf.parameters()) + list(self.qf_encoder.parameters())
        if self.twin_q:
            all_params += list(self.qf_twin.parameters()) + list(
                self.qf_twin_encoder.parameters()
            )
        for param in all_params:
            param.requires_grad = False
        output["q_curr"] = self.compute_q_values(q_batch_curr)
        for param in all_params:
            param.requires_grad = True
        
        # TD3: Add target policy smoothing (noise to target actions)
        target_noise = getattr(self.config, "target_noise", 0.2)
        if target_noise > 0:
            noise = torch.randn_like(actions_next) * target_noise
            noise_clip = getattr(self.config, "target_noise_clip", 0.5)
            noise = noise.clamp(-noise_clip, noise_clip)
            actions_next = (actions_next + noise).clamp(-1.0, 1.0)
        
        # Compute Q-values from target Q network
        q_batch_next = {
            Columns.OBS: batch[Columns.NEXT_OBS],
            Columns.ACTIONS: actions_next.detach(),
        }
        output["q_target_next"] = self.forward_target(q_batch_next).detach()
        
        # Store the target actions (for logging/debugging)
        # output["actions_next_target"] = actions_next.detach()
        
        return output
    
    @override(DefaultSACTorchRLModule)
    def compute_q_values(
        self, batch: Dict[str, Any], squeeze: bool = True
    ) -> Dict[str, Any]:
        """Compute Q-values which will be used for policy update"""
        # TD3 uses Q1 for policy update
        qvs = self._qf_forward_train_helper(
            batch, self.qf_encoder, self.qf, squeeze=squeeze
        )
        policy_update_mode = getattr(self.config,"policy_update_mode","q1")
        # Optionally use the mean, as in Fast TD3 
        if policy_update_mode == "q_mean":
            qvs = .5*(qvs+
            self._qf_forward_train_helper(batch, self.qf_twin_encoder, self.qf_twin, squeeze=squeeze)
            )
        # In SAC, we use the minimum of the two Q-values for policy update
        elif policy_update_mode == "q_min":
            qvs = torch.min(
                qvs,
                self._qf_forward_train_helper(
                    batch, self.qf_twin_encoder, self.qf_twin, squeeze=squeeze
                ),
            )
        return qvs
    

    @override(DefaultSACTorchRLModule)
    def _forward_inference(self, batch: Dict) -> Dict[str, Any]:
        """
        Inference forward pass.
        
        For TD3, we always use deterministic actions (no sampling).
        """
        output = {}
        
        # Pi encoder forward pass
        pi_encoder_outs = self.pi_encoder(batch)
        
        # Pi head
        action_logits = self.pi(pi_encoder_outs[ENCODER_OUT])
        
        # TD3: Always use deterministic actions (mean)
        # Create distribution and get mean
        action_dist_class = self.get_train_action_dist_cls()
        action_dist = action_dist_class(action_logits)
        
        # Return the mean (deterministic action)
        output[Columns.ACTIONS] = action_dist.sample()
        
        # Store dist inputs for compatibility
        output[Columns.ACTION_DIST_INPUTS] = action_logits
        
        return output



