# RLlib Implementation of TD3 

This directory contains an RLlib implementation of [TD3](https://arxiv.org/pdf/1802.09477), built as a lightweight wrapper of the [RLlib implementation of SAC](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#sac). TD3 and other training algorithms that were available in older versions of RLlib have been deprecated as the library has moved to a new API stack.

From an implementation prespective, TD3 differs from SAC primarily in that the learned policy is deterministic rather than stochastic. Changes are summarized below.



| Component | SAC | TD3 | File |
|-----------|-----|-----|-----------------|
| **Action Distribution** | `TorchSquashedGaussian`| `TorchDeterministic`| Catalog (`td3_catalog.py`)|
| **Action Sampling** | `action_dist.rsample()` | `action_dist.sample()` | RLModule (`td3_rl_module.py`) |
| **Entropy Terms** | In loss formula | None| Learner (`td3_torch_learner.py`) |
| **Alpha** | Adjusted via loss | None | Learner (`td3_torch_learner.py`) |
| **Policy Updates** | Every step | Every `policy_delay` steps | Learner (`td3_torch_learner.py`) |
| **Q function for Policy Update** | min(Q_1, Q_2) | Q_1 | RLModule (`td3_rl_module.py`) |
| **Target Actions** | Sample | Add noise (smoothing) | RLModule (`td3_rl_module.py`) |



## Summary of modifications

### Core Implementation
1. **`td3.py`** - Configuration, sets learner
2. **`td3_catalog.py`** - Sets distribution to be deterministic
2. **`td3_rl_module.py`** - Adds noise to sampled actions, sets Q network for policy loss function
3. **`td3_torch_learner.py`** - Removes entropy and alpha term from loss, adds policy update delay




## Quick Start

Run `pendulum_td3.py`
