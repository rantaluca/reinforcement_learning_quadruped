import gymnasium as gym
from stable_baselines3 import PPO
from envs.simple_quadruped_env import QuadrupedEnv

# Instantiating the environment
env = QuadrupedEnv()

# Loading the trained model
model_path = "models/ppo_quadruped.zip"
model = PPO.load(model_path, env=env, verbose=1)
print("Loaded model from", model_path)

# Running the model for 1000 steps
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Closing the env
env.close()
