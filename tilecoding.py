import numpy as np

class TileCoder:
    """
    Multi-tiling coarse coding for continuous states.
    - low, high: arrays with per-dimension min/max for normalization
    - bins: number of bins per dimension in each tiling
    - n_tilings: number of staggered tilings
    Returns sparse integer indices for active features.
    """
    def __init__(self, low, high, bins, n_tilings=8, seed=0):
        self.low  = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.bins = np.array(bins, dtype=np.int32)
        self.n_tilings = int(n_tilings)
        self.dim = len(self.bins)

        self.rng = np.random.default_rng(seed)
        # random fractional offsets in [0,1) per tiling and dim
        self.offsets = self.rng.uniform(0.0, 1.0, size=(self.n_tilings, self.dim)).astype(np.float32)

        self.features_per_tiling = int(np.prod(self.bins))
        self.total_features = self.n_tilings * self.features_per_tiling

        # strides for flattening multi-dim bin indices
        self.stride = np.cumprod([1] + list(self.bins[:-1])).astype(np.int32)

    def _to_unit(self, s):
        s = np.array(s, dtype=np.float32)
        return (s - self.low) / (self.high - self.low + 1e-8)

    def encode(self, s):
        """Return 1D integer indices of active features across all tilings."""
        u = np.clip(self._to_unit(s), 0.0, 1.0)
        idxs = np.empty(self.n_tilings, dtype=np.int32)
        for t in range(self.n_tilings):
            shifted = (u + self.offsets[t]) % 1.0
            b = np.minimum((shifted * self.bins).astype(np.int32), self.bins - 1)
            flat = int(np.dot(b, self.stride))
            idxs[t] = t * self.features_per_tiling + flat
        return idxs
