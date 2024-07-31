import gym
import pybullet
import pybullet_envs


import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env = gym.make("AntBulletEnv-v0")
env.render(mode="rgb_array")

MAX_AVERAGE_SCORE = 270

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[1024, 1024])]) #this is the policy network architecture

model = PPO("MlpPolicy", env, learning_rate= 0.0003,policy_kwargs=policy_kwargs, verbose=1)

for i in range(8000):
    print(f"Training iteration {i}")
    model.learn(total_timesteps=10000)
    model.save("ppo_ant")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)

del model