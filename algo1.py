import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import time

class QuadrupedEnv(gym.Env):
    def __init__(self):
        super(QuadrupedEnv, self).__init__()
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.planeId = p.loadURDF("plane.urdf")
        startPos = [0, 0, 0.2]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.boxId = p.loadURDF("spot_v1_description/urdf/spot_v1.urdf", startPos, startOrientation)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)  # 12 joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.planeId = p.loadURDF("plane.urdf")
        startPos = [0, 0, 0.2]  # Higher starting position to ensure time to hit the ground
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.boxId = p.loadURDF("spot_v1_description/urdf/spot_v1.urdf", startPos, startOrientation)
        p.setRealTimeSimulation(0)
        
        # Allow some time for the robot to stabilize
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Reset observation
        observation = self._get_observation().astype(np.float32)
        return observation, {}

    def step(self, action):
        for i in range(12):
            p.setJointMotorControl2(self.boxId, i, p.POSITION_CONTROL, targetPosition=action[i])

        p.stepSimulation()
        time.sleep(1./240.)

        observation = self._get_observation().astype(np.float32)
        reward = float(self._compute_reward(observation))
        terminated = self._is_done(observation)
        truncated = False  # Example: truncated is False because we are not using truncation in this example
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.boxId)
        lin_vel, ang_vel = p.getBaseVelocity(self.boxId)
        joint_states = p.getJointStates(self.boxId, range(12))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        # Concatenate all components to form the observation array
        observation = np.concatenate([pos, orn, lin_vel, ang_vel, joint_positions, joint_velocities])
        return observation

    def _compute_reward(self, observation):
        # Example reward function: reward for moving forward
        pos = observation[:3]
        reward = pos[0]
        return reward

    def _is_done(self, observation):
        # Example termination condition
        pos = observation[:3]
        if pos[2] < 0.2:  # if the robot falls
            return True
        return False

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

env = QuadrupedEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)  # Adjust timesteps as needed
model.save("ppo_quadruped")

# try the trained model

model = PPO.load("ppo_quadruped")
num_episodes = 5
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