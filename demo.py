import gym
import pybullet_envs
from stable_baselines3 import PPO

# Create the environment
env = gym.make("AntBulletEnv-v0")
env.render(mode="human")

# Load the trained model
model = PPO.load("ppo_ant")

# Number of episodes to test the model
num_episodes = 15

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render(mode="human")  # Render the environment
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()
