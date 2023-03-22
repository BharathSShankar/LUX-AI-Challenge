from typing import List
import flax.linen as nn
import jax.numpy as jnp
from neuralNets.convLayers import ResNetBlock

class MapScorer(nn.Module):
    map_size : int
    channels : List[int]
    n_layers : int

    @nn.compact
    def __call__(self, x: jnp.array):
        
        assert len(self.channels) >= self.n_layers

        for i in range(self.n_layers):
            x = ResNetBlock(self.channels[i], use_projection=True)(x)
            x = nn.LayerNorm()(x)
            x = nn.selu(x)
            x = nn.max_pool(x, (2,2))

        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.map_size * self.map_size)(x)
        x = nn.sigmoid(x)

        return x.reshape(self.map_size, self.map_size)