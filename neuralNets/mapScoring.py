from typing import List
import jax.numpy as jnp
from flax import linen as nn
from neuralNets.convLayers import ResNetBlock

class MapScorer(nn.Module):
    map_size: int
    channels: List[int]
    n_layers: int

    @nn.compact
    def __call__(self, x):
        # Ensure that the number of channels is greater than or equal to the number of layers
        assert len(self.channels) >= self.n_layers
        print(x.shape)
        x = jnp.swapaxes(x, 1, 3)
        # Apply the ResNetBlock, LayerNorm, SELU activation function, and max pooling for each layer
        for i in range(self.n_layers):
            x = ResNetBlock(self.channels[i], use_projection=True)(x)
            x = nn.LayerNorm()(x)
            x = nn.selu(x)
            x = nn.max_pool(x, (2,2))
            print(x.shape)
        # Reshape the output and apply the fully connected layer and sigmoid activation function
        x = nn.Conv(1, (1, 1))(x)
        print(x.shape)
        x = x.reshape(x.shape[-1], -1)
        x = nn.Dense(self.map_size * self.map_size)(x)
        x = nn.sigmoid(x)

        # Reshape the output to the final map size
        return jnp.reshape(x, (self.map_size, self.map_size))

