from typing import Any, Dict
import torch

from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID, TensorType

from ray.rllib.algorithms.sac.sac_learner import (
    ACTION_LOG_PROBS,
    ACTION_LOG_PROBS_NEXT,
    ACTION_PROBS,
    ACTION_PROBS_NEXT,
    LOGPS_KEY,
    QF_LOSS_KEY,
    QF_MAX_KEY,
    QF_MEAN_KEY,
    QF_MIN_KEY,
    QF_PREDS,
    QF_TARGET_NEXT,
    QF_TWIN_LOSS_KEY,
    QF_TWIN_PREDS,
    TD_ERROR_MEAN_KEY,
    SACLearner,
)
from ray.rllib.utils.metrics import ALL_MODULES, TD_ERROR_KEY


class TD3TorchLearner(SACTorchLearner):
    """
    TD3 Learner with minimal modifications to SAC.
    
    Changes from SAC:
    1. Critic loss: Remove entropy term
    2. Actor loss: Remove entropy term, add delayed updates
    3. Remove alpha (temperature) optimization
    """
    
    def build(self) -> None:
        super().build()
        # Track update counter for delayed policy updates
        self.update_counter = 0
    
    @override(SACTorchLearner)
    def configure_optimizers_for_module(
        self, module_id: ModuleID, config=None
    ) -> None:
        """Same as SAC but don't create alpha optimizer."""
        module = self._module[module_id]
        
        # Critic optimizer
        params_critic = self.get_parameters(module.qf_encoder) + self.get_parameters(
            module.qf
        )
        optim_critic = torch.optim.Adam(params_critic, eps=1e-7)
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="qf",
            optimizer=optim_critic,
            params=params_critic,
            lr_or_lr_schedule=config.critic_lr,
        )
        
        # Twin critic optimizer (TD3 always uses twin Q)
        if config.twin_q:
            params_twin_critic = self.get_parameters(
                module.qf_twin_encoder
            ) + self.get_parameters(module.qf_twin)
            optim_twin_critic = torch.optim.Adam(params_twin_critic, eps=1e-7)
            self.register_optimizer(
                module_id=module_id,
                optimizer_name="qf_twin",
                optimizer=optim_twin_critic,
                params=params_twin_critic,
                lr_or_lr_schedule=config.critic_lr,
            )
        
        # Actor optimizer
        params_actor = self.get_parameters(module.pi_encoder) + self.get_parameters(
            module.pi
        )
        optim_actor = torch.optim.Adam(params_actor, eps=1e-7)
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="policy",
            optimizer=optim_actor,
            params=params_actor,
            lr_or_lr_schedule=config.actor_lr,
        )
        
        # TD3: NO alpha optimizer (alpha is always 0)
    
    @override(SACTorchLearner)
    def _compute_loss_for_module_continuous(
        self,
        *,
        module_id: ModuleID,
        config: SACConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        """
        TD3 loss computation - minimal changes from SAC.
        
        Changes:
        1. actor_loss: Remove entropy term, add delayed updates
        2. Remove alpha_loss
        """
        
        # Get Q-values for the actually selected actions during rollout
        q_selected = fwd_out[QF_PREDS]
        q_twin_selected = fwd_out[QF_TWIN_PREDS]
        
        # Compute target Q-values with noisy actions
        q_target_next = fwd_out["q_target_next"]  # No entropy term!
        
        # Mask terminated states
        q_next_masked = (1.0 - batch[Columns.TERMINATEDS].float()) * q_target_next
        
        # Bellman target
        q_selected_target = (
            batch[Columns.REWARDS] + (config.gamma ** batch["n_step"]) * q_next_masked
        ).detach()
        
        # TD error (for prioritized replay)
        td_error = torch.abs(q_selected - q_selected_target)
        td_error += torch.abs(q_twin_selected - q_selected_target)
        td_error *= 0.5

        self.metrics.log_value(
            key=(module_id, TD_ERROR_KEY),
            value=td_error,
            reduce="item_series",
        )
        
        # Critic loss (Huber loss)
        critic_loss = torch.mean(
            batch["weights"]
            * torch.nn.HuberLoss(reduction="none", delta=1.0)(
                q_selected, q_selected_target
            )
        )
        critic_twin_loss = torch.mean(
            batch["weights"]
            * torch.nn.HuberLoss(reduction="none", delta=1.0)(
                q_twin_selected, q_selected_target
            )
        )
        
    


        if self.update_counter % config.policy_delay == 0:
            # TD3: Maximize Q(s, actor(s)) only
            actor_loss = -torch.mean(fwd_out["q_curr"])
        else:
            # Don't update actor
            actor_loss = torch.tensor(0.0, device=q_selected.device, requires_grad=True)
        
        # TD3: NO alpha loss (removed entirely)
        
        # Total loss
        #actor_loss = -torch.mean(fwd_out["q_curr"])
        total_loss = critic_loss + critic_twin_loss + actor_loss

        
        # Logging (simplified)
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: actor_loss,
                QF_LOSS_KEY: critic_loss,
                "alpha_loss": torch.tensor(0.0),
                "alpha_value": torch.tensor(1.0),
                "log_alpha_value": torch.tensor(0.0),
                "target_entropy": self.target_entropy[module_id],
                LOGPS_KEY: torch.mean(fwd_out["logp_resampled"]),
                QF_MEAN_KEY: torch.mean(fwd_out["q_curr"]),
                QF_MAX_KEY: torch.max(fwd_out["q_curr"]),
                QF_MIN_KEY: torch.min(fwd_out["q_curr"]),
                TD_ERROR_MEAN_KEY: torch.mean(td_error),
                QF_TWIN_LOSS_KEY: critic_twin_loss,
            },
            key=module_id,
            window=1,
        )
        

        
        self._temp_losses[(module_id, QF_LOSS_KEY)] = critic_loss
        self._temp_losses[(module_id, QF_TWIN_LOSS_KEY)] = critic_twin_loss
        self._temp_losses[(module_id, POLICY_LOSS_KEY)] = actor_loss
        
        # Update counter
        self.update_counter += 1
        
        return total_loss


