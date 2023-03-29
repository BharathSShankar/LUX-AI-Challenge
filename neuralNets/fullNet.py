from typing import List, Tuple
import flax.linen as nn
import jax.numpy as jnp
import jax
from neuralNets.attentionLayers import Att4Actions
from neuralNets.convLayers import ResNetBlock

from neuralNets.embeddings import EmbeddingLayer

class FullNetGame(nn.Module):
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
    def __call__(self, global_info: jnp.array, img_info: jnp.array, fact_vec: jnp.array, unit_vecs:jnp.array):

        global_emb = EmbeddingLayer(self.embed_dim)(global_info)
        global_emb = EmbeddingLayer(self.embed_dim * self.map_size * self.map_size)(global_emb)

        fact_emb = EmbeddingLayer(self.embed_dim)(fact_vec)
        fact_emb = EmbeddingLayer(self.embed_dim * 2)(fact_emb)

        unit_emb = EmbeddingLayer(self.embed_dim)(unit_vecs)
        unit_emb = EmbeddingLayer(self.embed_dim * 2)(unit_emb) 

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

        unit_actions, fact_actions = Att4Actions(60, 60)(globalFeatures, unit_emb, fact_emb)

        to_change = nn.sigmoid(nn.Dense(1)(unit_actions))
        
        unit_actions = nn.Dense(380)(unit_actions)
        fact_actions = nn.Dense(3)(fact_actions)

        unit_actions = unit_actions.reshape((unit_actions.shape[0], 20, 19)) 
        fact_actions = nn.softmax(fact_actions)
        return to_change, unit_actions, fact_actions