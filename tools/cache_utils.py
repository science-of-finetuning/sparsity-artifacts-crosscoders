from dictionary_learning.cache import PairedActivationCache, ActivationCache
import torch as th

class DifferenceCache(PairedActivationCache):
    def __init__(self, store_dir_1: str, store_dir_2: str):
        super().__init__(store_dir_1, store_dir_2)

    def __getitem__(self, index: int):
        return self.activation_cache_1[index] - self.activation_cache_2[index]


class DifferenceCache:
    def __init__(self, cache_1: ActivationCache, cache_2: ActivationCache):
        self.activation_cache_1 = cache_1
        self.activation_cache_2 = cache_2
        assert len(self.activation_cache_1) == len(self.activation_cache_2)

    def __len__(self):
        return len(self.activation_cache_1)

    def __getitem__(self, index: int):
        return self.activation_cache_1[index] - self.activation_cache_2[index]

    @property
    def tokens(self):
        return th.stack(
            (self.activation_cache_1.tokens, self.activation_cache_2.tokens), dim=0
        )

    @property
    def config(self):
        return self.activation_cache_1.config
