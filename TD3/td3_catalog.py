from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.core.distribution.distribution import Distribution
from ray.rllib.core.distribution.torch.torch_distribution import TorchDeterministic
from ray.rllib.utils.annotations import override

class TD3Catalog(SACCatalog):
    '''
    The catalog class used to build models for TD3.

    To be compatible with SAC framework, we will `sample' actions from a deterministic distribution.
    '''

    @override(SACCatalog)
    def get_action_dist_cls(self, framework: str) -> Distribution:
        """Action Distribution class is deterministic"""
        
        assert framework == "torch"

        return TorchDeterministic