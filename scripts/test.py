import gymnasium as gym
from stable_baselines3 import PPO
from envs.simple_quadruped_env import QuadrupedEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrapping the environment in DummyVecEnv for compatibility with Stable Baselines3
env = DummyVecEnv([lambda: QuadrupedEnv()])

# Loading the trained model
model_path = "models/ppo_quadruped_checkpoint_18940000_steps.zip"
model = PPO.load(model_path, env=env, verbose=1)
print("Loaded model from", model_path)

# Running the model for 1000 steps
obs = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, terminated, info = env.step(action)
    env.render()

# Closing the env
env.close()