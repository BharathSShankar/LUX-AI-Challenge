import jax.numpy as jnp
from flax import linen as nn
import jax

class EmbeddingLayer(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, input):
        # Apply a linear transformation to the input to reduce the dimensionality
        # of the input to self.embed_dim.
        x = nn.Dense(self.embed_dim)(input)
        
        # Apply Layer Normalization to normalize the activations of the previous layer
        # and improve training stability.
        x = nn.LayerNorm()(x)
        
        # Apply the CReLU activation function (which is simply the concatenation of
        # the ReLU and its negation) elementwise to the activations of the previous layer.
        x = jax.nn.celu(x)
        
        # Return the output activations.
        return x

