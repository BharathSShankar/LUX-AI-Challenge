import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

class Att4Actions(nn.Module):
    hidden_dim_unit : int
    hidden_dim_fact : int
    out_units: int = 19
    out_facts: int = 3
    
    @nn.compact
    def __call__(self, globalVec, localVecUnit, localVecFact):
        k_unit = nn.Dense(self.hidden_dim_unit)(globalVec)
        v_unit = nn.Dense(self.hidden_dim_unit)(globalVec)
        
        k_fact = nn.Dense(self.hidden_dim_fact)(globalVec)
        v_fact = nn.Dense(self.hidden_dim_fact)(globalVec) 

        q_unit = nn.Dense(self.hidden_dim_unit)(localVecUnit)
        q_fact = nn.Dense(self.hidden_dim_fact)(localVecFact)

        unit_actions = nn.dot_product_attention(q_unit, k_unit, v_unit)
        fact_actions = nn.dot_product_attention(q_fact, k_fact, v_fact) 

        unit_actions = nn.Dense(self.out_units)(unit_actions)
        fact_actions = nn.Dense(self.out_facts)(fact_actions)

        return  unit_actions, fact_actions