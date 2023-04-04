import luxai_s2
import jux
from agentHelpers.agent_wrappers import JuxWrapperEnv
from neuralNets.actorNet import ActorNet
from neuralNets.criticNet import CriticNet
import jax
import jax.numpy as jnp
import optax 

def get_next_key(key):
    new_key, sub = jax.random.split(key)
    return sub

key = jax.random.PRNGKey(0)

global_info_shape = (33,)
img_feat_shape = (48, 48, 37)
unit_shape = (300, 7)
fact_shape = (40, 7)

actorNet = ActorNet(
    3,
    [32, 64, 256], 
    [3, 1, 1], 
    [True, True, True],
    256,
    128,
    160,
    120
)

actor_params = actorNet.init(
    get_next_key(key),
    jax.random.normal(key, global_info_shape),
    jax.random.normal(key, img_feat_shape),
    jax.random.normal(key, unit_shape),
    jax.random.normal(key, fact_shape),
)

criticNet = CriticNet(
    3,
    [32, 64, 256],  
    [3, 1, 1],
    True,
    256, 
    128, 
    160, 
    120
)

global_info_shape = (33,)
img_feat_shape = (48, 48, 37)

critic_params = criticNet.init(
    get_next_key(key),
    jax.random.normal(key, global_info_shape),
    jax.random.normal(key, img_feat_shape),
)
