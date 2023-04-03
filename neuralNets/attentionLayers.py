import jax.numpy as jnp
from flax import linen as nn

class Att4Actions(nn.Module):
    hidden_dim_unit: int
    hidden_dim_fact: int
    out_units: int = 19
    out_facts: int = 3

    @nn.compact
    def __call__(self, globalVec, localVecUnit, localVecFact):
        # Apply linear transformations to global vector for keys and values
        k_unit = nn.Dense(self.hidden_dim_unit)(globalVec)
        v_unit = nn.Dense(self.hidden_dim_unit)(globalVec)

        # Apply linear transformations to global vector for keys and values
        k_fact = nn.Dense(self.hidden_dim_fact)(globalVec)
        v_fact = nn.Dense(self.hidden_dim_fact)(globalVec)

        # Apply linear transformations to local vectors for queries
        q_unit = nn.Dense(self.hidden_dim_unit)(localVecUnit)
        q_fact = nn.Dense(self.hidden_dim_fact)(localVecFact)

        # Compute dot product attention between local vectors and global vectors for units
        unit_actions = nn.attention.dot_product_attention(q_unit, k_unit, v_unit)

        # Compute dot product attention between local vectors and global vectors for facts
        fact_actions = nn.attention.dot_product_attention(q_fact, k_fact, v_fact)

        # Apply linear transformation to output of unit attention layer
        unit_actions = nn.Dense(self.out_units)(unit_actions)
        unit_actions = jnp.selu(unit_actions)

        # Apply linear transformation to output of fact attention layer
        fact_actions = nn.Dense(self.out_facts)(fact_actions)
        fact_actions = jnp.selu(fact_actions)

        # Return the outputs
        return unit_actions, fact_actions

