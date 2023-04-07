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
img_feat_shape = (48, 48, 31)
unit_shape = (300, 7)
fact_shape = (40, 8)

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
    jax.random.normal(key, fact_shape),
    jax.random.normal(key, unit_shape),
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
img_feat_shape = (48, 48, 31)

memory_0 = PPOMemory(64)
memory_1 = PPOMemory(64)
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

scheduler = optax.cosine_decay_schedule(
    0.0001, decay_steps=100000000, alpha=0.95)

opt_act = optax.adam(
    3e-4
)

opt_crit = optax.adam(
    3e-4
)



actor_state = opt_act.init(actor_params)
critic_state = opt_crit.init(critic_params)

player_0_trainer = nnAgentTrainer(
    "player_0",
    EnvConfig(),
    controller,
    actorNet,
    criticNet,
    memory_0,
    actor_params,
    critic_params,
    opt_act,
    opt_crit,
    actor_state,
    critic_state,
)

player_1_trainer = nnAgentTrainer(
    "player_1",
    EnvConfig(),
    controller,
    actorNet,
    criticNet,
    memory_1,
    actor_params,
    critic_params,
    opt_act,
    opt_crit,
    actor_state,
    critic_state,
)
print("Setup Complete!")

for i in tqdm(range(200)):
    state = jux_env.reset(seed=jax.random.randint(
        key, (), minval=0, maxval=200000000))
    key = get_next_key(key)
    lux_state = state.to_lux()
    bid0 = player_0_trainer.bid_policy(0, lux_state)
    bid1 = player_1_trainer.bid_policy(0, lux_state)
    bid = dict(player_0=bid0, player_1=bid1)
    state, observations, rewards, dones, infos = jux_env.step(0,  bid, state)
    while state.real_env_steps < 0:
        lux_state = state.to_lux()
        action0 = player_0_trainer.factory_placement_policy(
            state.real_env_steps, lux_state)
        action1 = player_1_trainer.factory_placement_policy(
            state.real_env_steps, lux_state)
        action = dict(player_0=action0, player_1=action1)
        state, observations, rewards, dones, infos = jux_env.step(
            state.real_env_steps, action, state)
    while not dones[0]:
        lux_state = state.to_lux()
        actions_0, act_probs_0, value_0, excess_params_0 = player_0_trainer.choose_act(
            state.real_env_steps, lux_state)
        actions_1, act_probs_1, value_1, excess_params_1 = player_1_trainer.choose_act(
            state.real_env_steps, lux_state)
        action = dict(player_0=actions_0, player_1=actions_1)
        state, observations, rewards, dones, infos = jux_env.step(
            state.real_env_steps + 1, action, state)
        
        player, gameState, to_change, unit_actions_logits, fact_actions_logits, unit_actions_disc_params_R, \
            unit_actions_disc_params_Rep, unit_actions_disc_params_N, global_info, img_info, fact_obs, unit_obs, avail_facts, avail_units = excess_params_0
        player_0_trainer.memory.store_memory(
            to_change, global_info, img_info, fact_obs, unit_obs,\
            actions_0, actions_1, fact_actions_logits, unit_actions_logits, \
            unit_actions_disc_params_R, unit_actions_disc_params_N, unit_actions_disc_params_Rep,
            value_0, rewards[0], dones[0], avail_facts, avail_units
        )

        player, gameState, to_change, unit_actions_logits, fact_actions_logits, unit_actions_disc_params_R, \
            unit_actions_disc_params_Rep, unit_actions_disc_params_N, global_info, img_info, fact_obs, unit_obs, avail_facts, avail_units = excess_params_1
        player_1_trainer.memory.store_memory(
            to_change, global_info, img_info, fact_obs, unit_obs,\
            actions_0, actions_1, fact_actions_logits, unit_actions_logits, \
            unit_actions_disc_params_R, unit_actions_disc_params_N, unit_actions_disc_params_Rep,
            value_1, rewards[1], dones[1], avail_facts, avail_units
        )

        if state.real_env_steps  % 128 == 0 and state.real_env_steps > 0:
            player_1_trainer.actor_params, player_1_trainer.actor_state, \
            player_1_trainer.critic_params, player_1_trainer.critic_state = player_0_trainer.learn(5, 0.9, 0.95, 0.2)
            player_0_trainer.actor_params, player_0_trainer.actor_state, \
            player_0_trainer.critic_params, player_0_trainer.critic_state = player_1_trainer.learn(5, 0.9, 0.95, 0.2) 