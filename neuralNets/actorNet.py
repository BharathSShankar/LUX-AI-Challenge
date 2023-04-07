from typing import List, Tuple
import jax.numpy as jnp
from flax import linen as nn

# Importing custom modules
from neuralNets.attentionLayers import Att4Actions
from neuralNets.convLayers import ResNetBlock
from neuralNets.embeddings import EmbeddingLayer


class ActorNet(nn.Module):
    num_resid_layers: int
    channel_nums: List[int]
    k_sizes: List[int]
    use_se: List[bool]
    hidden_dim_unit: int
    hidden_dim_fact: int
    embed_dim: int
    global_feat: int
    strides: Tuple[int, int] = (1, 1)
    map_size: int = 48

    @nn.compact
    def __call__(self, global_info: jnp.array, img_info: jnp.array, fact_vec: jnp.array, unit_vecs:jnp.array):
        # Function to call the FullNetGame module with inputs and outputs

        # Embedding the global information using EmbeddingLayer
        global_emb = EmbeddingLayer(self.embed_dim)(global_info)
        global_emb = EmbeddingLayer(self.embed_dim * self.map_size * self.map_size)(global_emb)

        # Embedding the fact vectors using EmbeddingLayer
        fact_emb = EmbeddingLayer(self.embed_dim)(fact_vec)
        fact_emb = EmbeddingLayer(self.embed_dim * 2)(fact_emb)

        # Embedding the unit vectors using EmbeddingLayer
        unit_emb = EmbeddingLayer(self.embed_dim)(unit_vecs)
        unit_emb = EmbeddingLayer(self.embed_dim * 2)(unit_emb) 

        # Reshaping the global embeddings
        global_emb = global_emb.reshape((self.map_size, self.map_size, -1))
        img_info = img_info.reshape(self.map_size, self.map_size, -1)
        # Concatenating image information and global embeddings
        img_features = jnp.concatenate((img_info, global_emb), axis=2)
        
        # Passing image information through residual blocks
        for i in range(self.num_resid_layers):
            img_features = ResNetBlock(
                self.channel_nums[i],
                use_se=True,
                strides=self.strides,
                ksize=self.k_sizes[i]
                )(img_features)
            img_features = nn.max_pool(img_features, (2, 2))
        
        # Reshaping image features
        globalFeatures = img_features.reshape((img_features.shape[-1], -1))

        # Getting unit actions and fact actions using Att4Actions
        unit_actions, fact_actions = Att4Actions(60, 60)(globalFeatures, unit_emb, fact_emb)

        # Getting probability of unit actions to change
        to_change = nn.sigmoid(nn.Dense(1)(unit_actions))

        # Getting logits for unit actions and fact actions
        unit_actions_logits = nn.Dense(260)(unit_actions)
        fact_actions_logits = nn.Dense(3)(fact_actions)

        # Getting continuous and discrete parameters for unit actions
        unit_actions_disc_params_R = nn.Dense(100)(unit_actions)
        unit_actions_disc_params_N = nn.Dense(160)(unit_actions)
        unit_actions_disc_params_Rep = nn.Dense(80)(unit_actions)

        unit_actions_logits = unit_actions_logits.reshape((unit_actions_logits.shape[0], 20, 13))

        unit_actions_disc_params_R = unit_actions_disc_params_R.reshape((unit_actions_logits.shape[0], 20, 5))
        unit_actions_disc_params_N = unit_actions_disc_params_N.reshape((unit_actions_logits.shape[0], 20, 8))
        unit_actions_disc_params_Rep = unit_actions_disc_params_Rep.reshape((unit_actions_logits.shape[0], 20, 4))

        return to_change, unit_actions_logits, fact_actions_logits, unit_actions_disc_params_R, unit_actions_disc_params_Rep, unit_actions_disc_params_N