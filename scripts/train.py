import os
import gymnasium as gym
from stable_baselines3 import PPO
from envs.simple_quadruped_env import QuadrupedEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Creating the environment
env = QuadrupedEnv()

# Model path
model_save_path = "models/ppo_quadruped"
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_quadruped_checkpoint')

# Checking if a model exists
if os.path.exists(model_save_path + ".zip"):
    model = PPO.load(model_save_path, env=env, verbose=1)
    print("Loaded existing model from", model_save_path)
else:
    model = PPO("MlpPolicy", env, verbose=1)
    print("Created a new model.")

try:
    # Training the model 
    model.learn(total_timesteps=1000000000, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("Training interrupted. Saving the model.")
    model.save(model_save_path)


# Saving the trained model
model.save(model_save_path)
print("Model saved at", model_save_path)

# Closing the env 
env.close()

