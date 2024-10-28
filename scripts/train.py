import os
import gymnasium as gym
from stable_baselines3 import PPO
from envs.simple_quadruped_env import QuadrupedEnv

# Creating the environment
env = QuadrupedEnv()

# Model path
model_save_path = "models/ppo_quadruped"

# Checking if a model exists
if os.path.exists(model_save_path + ".zip"):
    model = PPO.load(model_save_path, env=env, verbose=1)
    print("Loaded existing model from", model_save_path)
else:
    model = PPO("MlpPolicy", env, verbose=1)
    print("Created a new model.")

# Training the model
model.learn(total_timesteps=100000)

# Saving the trained model
model.save(model_save_path)
print("Model saved at", model_save_path)

# Closing the env 
env.close()
