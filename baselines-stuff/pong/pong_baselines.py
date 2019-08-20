import gym
from stable_baselines import ACER
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv

# trying to get an idea of how quickly my computer can train this
pong_env = gym.make('Pong-v0')
pong_env = DummyVecEnv([lambda: pong_env])
pong_model_acer = ACER(CnnPolicy, pong_env, verbose=0,
                       tensorboard_log="./../../data/baselines-stuff/pong/acer_pong_tensorboard/")
pong_model_acer.learn(total_timesteps=50_000_000, tb_log_name="run-1-50_000_000")

# since I know I'll be stopping it early
pong_model_acer.save('./../../data/baselines-stuff/pong/terrible_pong_model_acer')