import jax.numpy as jnp
from jax import nn
from flax import linen as nn
from typing import List, Tuple

from neuralNets.attentionLayers import Att4Actions
from neuralNets.convLayers import ResNetBlock
from neuralNets.embeddings import EmbeddingLayer

class CriticNet(nn.Module):
    num_resid_layers: int
    channel_nums: List[int]
    k_sizes: List[int]
    use_se: List[bool]
    hidden_dim_unit: int
    hidden_dim_fact: int
    embed_dim: int
    global_feat: int
    strides: Tuple[int] = (1, 1)
    map_size: int = 48

    @nn.compact
    def __call__(self,
                 global_info: jnp.array,
                 img_info: jnp.array):

        # Embed global info into a vector and reshape it into an image
        global_emb = EmbeddingLayer(self.embed_dim)(global_info)
        global_emb = EmbeddingLayer(self.embed_dim * self.map_size * self.map_size)(global_emb)
        global_emb = global_emb.reshape((self.map_size, self.map_size, -1))

        # Concatenate the image information with the global information
        img_features = jnp.concatenate((img_info, global_emb), axis= -1)

        # Apply a ResNetBlock followed by max pooling for each residual layer
        for i in range(self.num_resid_layers):
            img_features = ResNetBlock(self.channel_nums[i],
                                       use_se=True,
                                       strides=self.strides,
                                       ksize=self.k_sizes[i])(img_features)
            img_features = nn.max_pool(img_features, window_shape=(2, 2))

        # Reshape the image into global features
        global_features = img_features.reshape((img_features.shape[-1], -1))

        # Apply two fully connected layers followed by layer normalization and ReLU activation
        value = nn.Dense(self.embed_dim // 4)(global_features)
        value = nn.LayerNorm()(value)
        value = nn.relu(value)

        value = nn.Dense(self.embed_dim // 16)(value)
        value = nn.LayerNorm()(value)
        value = nn.relu(value)

        # Compute the critic value and apply tanh activation
        value = nn.Dense(1)(value)
        value = nn.tanh(value)

        return value

