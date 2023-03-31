from typing import List, Tuple
import flax.linen as nn
import jax.numpy as jnp
import jax
from neuralNets.attentionLayers import Att4Actions
from neuralNets.convLayers import ResNetBlock

from neuralNets.embeddings import EmbeddingLayer

class CriticGame(nn.Module):
    num_resid_layers:int
    channel_nums: List[int]
    k_sizes: List[int]
    use_se: List[bool]
    hidden_dim_unit: int
    hidden_dim_fact: int
    strides: Tuple[int, int] = (1, 1)
    embed_dim : int
    global_feat: int
    map_size : int = 48

    @nn.compact
    def __call__(self, global_info: jnp.array, img_info: jnp.array):

        global_emb = EmbeddingLayer(self.embed_dim)(global_info)
        global_emb = EmbeddingLayer(self.embed_dim * self.map_size * self.map_size)(global_emb)

        global_emb = global_emb.reshape((-1, self.map_size, self.map_size))

        img_features = jax.lax.concatenate((img_info, global_emb), dimension=0)

        for i in range(self.num_resid_layers):
            img_features = ResNetBlock(
                self.channels_nums[i],
                use_se=True,
                strides = self.strides,
                ksize = self.k_sizes[i]
                )(img_features)
            img_features = nn.max_pool(img_features, (2, 2))
        
        globalFeatures = img_features.reshape((img_features.shape[0], -1))

        value = nn.Dense(self.embed_dim // 4)(globalFeatures)
        value = nn.LayerNorm()(value)
        value = nn.relu(value)
        
        value = nn.Dense(self.embed_dim // 16)(value)
        value = nn.LayerNorm()(value)
        value = nn.relu(value)

        value = nn.Dense(1)(value)
        value = nn.tanh(value) 
        
        return  value