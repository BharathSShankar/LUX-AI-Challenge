import haiku as hk
import jax.numpy as jnp

class Att4Actions(hk.Module):
    def __init__(self, hidden_dim_unit, hidden_dim_fact, out_units=19, out_facts=3):
        super().__init__()
        self.hidden_dim_unit = hidden_dim_unit
        self.hidden_dim_fact = hidden_dim_fact
        self.out_units = out_units
        self.out_facts = out_facts

    def __call__(self, globalVec, localVecUnit, localVecFact):
        # Apply linear transformations to global vector for keys and values
        k_unit = hk.Linear(self.hidden_dim_unit)(globalVec)
        v_unit = hk.Linear(self.hidden_dim_unit)(globalVec)

        # Apply linear transformations to global vector for keys and values
        k_fact = hk.Linear(self.hidden_dim_fact)(globalVec)
        v_fact = hk.Linear(self.hidden_dim_fact)(globalVec)

        # Apply linear transformations to local vectors for queries
        q_unit = hk.Linear(self.hidden_dim_unit)(localVecUnit)
        q_fact = hk.Linear(self.hidden_dim_fact)(localVecFact)

        # Compute dot product attention between local vectors and global vectors for units
        unit_actions = hk.attention.dot_product_attention(q_unit, k_unit, v_unit)

        # Compute dot product attention between local vectors and global vectors for facts
        fact_actions = hk.attention.dot_product_attention(q_fact, k_fact, v_fact)

        # Apply linear transformation to output of unit attention layer
        unit_actions = hk.Linear(self.out_units)(unit_actions)

        # Apply linear transformation to output of fact attention layer
        fact_actions = hk.Linear(self.out_facts)(fact_actions)

        # Return the outputs
        return unit_actions, fact_actions
