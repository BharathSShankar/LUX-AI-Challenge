import jax.numpy as jnp
import flax.linen as nn

class EmbeddingLayer(nn.Module):
    embed_dim : int

    @nn.compact
    def __call__(self, input):
        x = nn.Dense(self.embed_dim)(input)
        x = nn.LayerNorm()(x)
        x = nn.celu(x)
        return x