import haiku as hk
import jax.numpy as jnp
import jax

class EmbeddingLayer(hk.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def __call__(self, input):
        # Apply a linear transformation to the input to reduce the dimensionality
        # of the input to self.embed_dim.
        x = hk.Linear(self.embed_dim)(input)
        
        # Apply Layer Normalization to normalize the activations of the previous layer
        # and improve training stability.
        x = hk.LayerNorm()(x)
        
        # Apply the CReLU activation function (which is simply the concatenation of
        # the ReLU and its negation) elementwise to the activations of the previous layer.
        x = jax.nn.celu(x)
        
        # Return the output activations.
        return x
