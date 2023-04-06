import luxai_s2
import jux
from agentHelpers.agent_controller import OverallController
from agentHelpers.agent_wrappers import JuxWrapperEnv
from neuralNets.PPOMemory import PPOMemory
from neuralNets.actorNet import ActorNet
from neuralNets.criticNet import CriticNet
from agentHelpers.agent_constants import REW_WTS
import jax
import jax.numpy as jnp
import optax
from jux.env import JuxEnv
from jux.config import JuxBufferConfig, EnvConfig
from nnAgent import nnAgentTrainer 
from tqdm import tqdm

n_iters = 2000

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
print("Initializing ActorNet....")
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

memory = PPOMemory(64)
print("Initialising CriticNet......")
critic_params = criticNet.init(
    get_next_key(key),
    jax.random.normal(key, global_info_shape),
    jax.random.normal(key, img_feat_shape),
)

jux_env = JuxEnv(
    env_cfg=EnvConfig(),
    buf_cfg=JuxBufferConfig(MAX_N_UNITS=200),
)

jux_env = JuxWrapperEnv(jux_env, REW_WTS)
state = jux_env.reset(0)

controller = OverallController(EnvConfig())

scheduler = cosine_decay_scheduler = optax.cosine_decay_schedule(0.0001, decay_steps=100000000, alpha=0.95)

opt = optax.adam(
    scheduler
)

actor_state = opt.init(actor_params)
critic_state = opt.init(critic_params)

player_0_trainer = nnAgentTrainer(
    "player_0",
    EnvConfig(),
    controller,
    actorNet,
    criticNet,
    memory,
    actor_params, 
    critic_params, 
    opt,
    opt,
    actor_state,
    critic_state,
)

player_1_trainer = nnAgentTrainer(
    "player_1",
    EnvConfig(),
    controller,
    actorNet,
    criticNet,
    memory,
    actor_params, 
    critic_params, 
    opt,
    opt,
    actor_state,
    critic_state,
)
print("Setup Complete!")

for i in tqdm(range(2)):
    state = jux_env.reset(seed = jax.random.randint(key, (), minval=0, maxval=200000000))
    key = get_next_key(key)
    lux_state = state.to_lux()
    bid0 = player_0_trainer.bid_policy(0, lux_state)
    bid1 = player_1_trainer.bid_policy(0, lux_state)
    bid = dict(player_0 = bid0, player_1 = bid1)
    state, observations, rewards, dones, infos = jux_env.step(0,  bid, state)
    while state.real_env_steps < 0:
        lux_state = state.to_lux()
        action0 = player_0_trainer.factory_placement_policy(state.real_env_steps, lux_state)
        action1 = player_1_trainer.factory_placement_policy(state.real_env_steps, lux_state) 
        action = dict(player_0 = action0, player_1 = action1)
        print(action)
        state, observations, rewards, dones, infos = jux_env.step(state.real_env_steps, action, state)
    print(state.to_lux())