from jux.env import JuxEnv, JuxAction
from jux.config import JuxBufferConfig, EnvConfig

from agentHelpers.agent_obs_processing import global_2_vec
from agentHelpers.agent_wrappers import LuxObsWrapper
from lux.kit import obs_to_game_state

jux_env = JuxEnv(
    env_cfg=EnvConfig(),
    buf_cfg=JuxBufferConfig(MAX_N_UNITS=200),
)

jux_env.buf_cfg
state = jux_env.reset(seed=0)

import jux.utils

lux_env, lux_actions = jux.utils.load_replay(f'https://www.kaggleusercontent.com/episodes/{46215591}.json')

jux_env, state = JuxEnv.from_lux(lux_env, buf_cfg=JuxBufferConfig(MAX_N_UNITS=200))

lux_act = next(lux_actions)


bid, faction = jux.actions.bid_action_from_lux(lux_act)

state, (observations, rewards, dones, infos) = jux_env.step_bid(state, bid, faction)

while state.real_env_steps < 0:
    lux_act = next(lux_actions)
    spawn, water, metal = jux.actions.factory_placement_action_from_lux(lux_act)
    state, (observations, rewards, dones, infos) = jux_env.step_factory_placement(state, spawn, water, metal)
for i in range(1000):
    lux_act = next(lux_actions)
    jux_act = JuxAction.from_lux(state, lux_act)

# step
    state, (observations, rewards, dones, infos) = jux_env.step_late_game(state, jux_act)
    print(observations)