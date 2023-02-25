import os
import gym
from stable_baselines3 import A2C
# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html

model_dir = "SB3/models/A2C"
log_dir = "SB3/logs/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = gym.make("LunarLander-v2")
env.reset()

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
TIMESTEPS = 10000
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name="A2C")
    model.save(f"{model_dir}/A2C_{i * TIMESTEPS}")