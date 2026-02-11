from typing import Optional, Type, Union
from typing_extensions import Self

from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.sac import SAC, SACConfig
from td3_torch_learner import TD3TorchLearner
from ray.rllib.utils.annotations import override

class TD3Config(SACConfig):
    """TD3 Configuration - just extends SAC with TD3-specific parameters."""
    
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or TD3)
        
        # Dont ever actually use alpha
        self.initial_alpha = 1.0
        self.alpha_lr = 0.0  # Don't learn alpha
        
        # TD3-specific parameters
        self.policy_delay = 2  # Update policy every N critic updates
        self.target_noise = 0.2  # Target policy smoothing noise std
        self.target_noise_clip = 0.5  # Clip noise to this range

        self.policy_update_mode = "q1" ## q1, q_mean, q_min

    
    def training(
        self,
        *,
        policy_delay: Optional[int] = NotProvided,
        target_noise: Optional[float] = NotProvided,
        target_noise_clip: Optional[float] = NotProvided,
        policy_update_mode: Optional[str] = NotProvided,
        **kwargs
    ) -> Self:
        """Extends SAC training config with additional TD3-specific parameters.
        
        Args:  
            policy_delay: Update policy every N critic updates
            target_noise: Target policy smoothing noise std
            target_noise_clip: Clip noise to this range
            policy_update_mode: "q1", "q_mean", or "q_min"; default is "q1"
                - "q1": Use Q1 for policy update
                - "q_mean": Use the mean of Q1 and Q2 for policy update
                - "q_min": Use the minimum of Q1 and Q2 for policy update
        """
        super().training(**kwargs)
        
        if policy_delay is not NotProvided:
            self.policy_delay = policy_delay
        if target_noise is not NotProvided:
            self.target_noise = target_noise
        if target_noise_clip is not NotProvided:
            self.target_noise_clip = target_noise_clip
        if policy_update_mode is not NotProvided:
            self.policy_update_mode = policy_update_mode
        
        # Dont ever actually use alpha
        self.initial_alpha = 1.0
        self.alpha_lr = 0.0
        
        return self
    
    @override(SACConfig)
    def get_default_learner_class(self) -> Union[Type["Learner"], str]:
        if self.framework_str == "torch":
            return TD3TorchLearner
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. Use `torch`."
            )
    
    @property
    def _model_config_auto_includes(self):
        """Include TD3-specific parameters in model_config so RLModule can access them."""
        return super()._model_config_auto_includes | {
            "target_noise": self.target_noise,
            "target_noise_clip": self.target_noise_clip,
            "policy_delay": self.policy_delay,
            "policy_update_mode": self.policy_update_mode,
        }


class TD3(SAC):
    """
    Twin Delayed DDPG (TD3) - Minimal wrapper around SAC.
    """
    
    @classmethod
    def get_default_config(cls) -> TD3Config:
        return TD3Config()
