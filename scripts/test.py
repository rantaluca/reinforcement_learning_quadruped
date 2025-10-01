import gymnasium as gym
import os
from stable_baselines3 import PPO
from envs.simple_quadruped_env import QuadrupedEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrapping the environment in DummyVecEnv for compatibility with Stable Baselines3
env = DummyVecEnv([lambda: QuadrupedEnv()])

# Loading the newest model in the folder trained model
model_folder = "models"
# find the newest model in the folder
model_files = [f for f in os.listdir(model_folder) if f.startswith("ppo_quadruped") and f.endswith(".zip")]
model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_folder, x)), reverse=True)
if not model_files:
    raise FileNotFoundError("No model files found in the models folder.")
model_path = os.path.join(model_folder, model_files[0]) 
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