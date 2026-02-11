"""
Simple TD3 on Pendulum - using TD3 as a lightweight wrapper around SAC

This shows how to use the TD3RLModule to make SAC's policy deterministic.
"""

from torch import nn
from td3 import TD3Config
from td3_rl_module import TD3TorchRLModule
from ray.rllib.algorithms.sac.sac import SACConfig

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.examples.utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)

parser = add_rllib_example_script_args(
    default_timesteps=20000,
    default_reward=-250.0,
)
args = parser.parse_args()

##compare to running sac


config = (
    SACConfig()
    .environment("Pendulum-v1")
    .training(
        initial_alpha=1.001,
        # Use a smaller learning rate for the policy.
        actor_lr=2e-4 * (args.num_learners or 1) ** 0.5,
        critic_lr=8e-4 * (args.num_learners or 1) ** 0.5,
        alpha_lr=9e-4 * (args.num_learners or 1) ** 0.5,
        # TODO (sven): Maybe go back to making this a dict of the sub-learning rates?
        lr=None,
        target_entropy="auto",
        n_step=(2, 5),
        tau=0.005,
        train_batch_size_per_learner=256,
        target_network_update_freq=1,
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 100000,
            "alpha": 1.0,
            "beta": 0.0,
        },
        num_steps_sampled_before_learning_starts=256 * (args.num_learners or 1),
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="relu",
            fcnet_kernel_initializer=nn.init.xavier_uniform_,
            head_fcnet_hiddens=[],
            head_fcnet_activation=None,
            head_fcnet_kernel_initializer="orthogonal_",
            head_fcnet_kernel_initializer_kwargs={"gain": 0.01},
        ),
    )
    .reporting(
        metrics_num_episodes_for_smoothing=5,
    )
)
"# Build the SAC algo object from the config and run 1 training iteration."
algo = config.build()
for i in range(5):
    result = algo.train()
    print(f"Iter {i}: {result['env_runners']['episode_return_mean']:.2f}")

###use td3 now

config = (
    TD3Config()
    .environment("Pendulum-v1")
    .training(
        initial_alpha=1.001,
        # Use a smaller learning rate for the policy.
        actor_lr=2e-4 * (args.num_learners or 1) ** 0.5,
        critic_lr=8e-4 * (args.num_learners or 1) ** 0.5,
        alpha_lr=9e-4 * (args.num_learners or 1) ** 0.5,
        # TODO (sven): Maybe go back to making this a dict of the sub-learning rates?
        lr=None,
        target_entropy="auto",
        n_step=(2, 5),
        tau=0.005,
        train_batch_size_per_learner=256,
        target_network_update_freq=1,
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 100000,
            "alpha": 1.0,
            "beta": 0.0,
        },
        num_steps_sampled_before_learning_starts=256 * (args.num_learners or 1),
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="relu",
            fcnet_kernel_initializer=nn.init.xavier_uniform_,
            head_fcnet_hiddens=[],
            head_fcnet_activation=None,
            head_fcnet_kernel_initializer="orthogonal_",
            head_fcnet_kernel_initializer_kwargs={"gain": 0.01},
        ),
    )
    .reporting(
        metrics_num_episodes_for_smoothing=5,
    )
)


    
config.policy_delay = 1
config.target_noise = 0.2
config.target_noise_clip = 0.5


algo = config.build()
for i in range(5):
    result = algo.train()
    print(f"Iter {i}: {result['env_runners']['episode_return_mean']:.2f}")

# if __name__ == "__main__":
#     run_rllib_example_script_experiment(algo, args)
