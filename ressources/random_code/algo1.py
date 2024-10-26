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
    def __init__(self, render=True):
        super(QuadrupedEnv, self).__init__()
        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.planeId = p.loadURDF("plane.urdf")
        startPos = [0, 0, 0.5]  # Start at some height
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.boxId = p.loadURDF("spot_v1_description/urdf/spot_v1.urdf", useFixedBase=False,
                                basePosition=startPos, baseOrientation=startOrientation)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)  # 12 joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.planeId = p.loadURDF("plane.urdf")
        
        startPos = [0, 0, 0.5]  # Start at some height
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.boxId = p.loadURDF("spot_v1_description/urdf/spot_v1.urdf", useFixedBase=False,
                                basePosition=startPos, baseOrientation=startOrientation)
        
        # Set joints to a neutral starting position
        for i in range(12):
            p.resetJointState(self.boxId, i, targetValue=0)
        
        p.setRealTimeSimulation(0)
        
        # Step simulation for stabilization
        for _ in range(100):
            p.stepSimulation()
            # time.sleep(1./240.)  # Remove sleep during training
                
        observation = self._get_observation().astype(np.float32)
        return observation, {}

    def step(self, action):
        for i in range(12):
            p.setJointMotorControl2(self.boxId, i, p.POSITION_CONTROL, targetPosition=action[i])

        p.stepSimulation()
        # time.sleep(1./240.)  # Remove sleep during training

        observation = self._get_observation().astype(np.float32)
        reward = float(self._compute_reward(observation))
        terminated = self._is_done(observation)
        truncated = False  # No truncation in this example
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
        # Reward for moving forward along the x-axis
        pos = observation[:3]
        reward = pos[0]
        return reward

    def _is_done(self, observation):
        # Terminate if the robot falls
        pos = observation[:3]
        if pos[2] < 0.2:  # if the robot falls
            return True
        return False

    def render(self, mode='human'):
        pass  # Rendering is handled by PyBullet GUI

    def close(self):
        p.disconnect()

# Training the model
env = QuadrupedEnv(render=False)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)  # Adjust timesteps as needed
model.save("ppo_quadruped")

# Evaluation
env = QuadrupedEnv(render=True)  # Enable rendering for evaluation
model = PPO.load("ppo_quadruped")
num_episodes = 5
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # env.render(mode="human")  # Not necessary; PyBullet GUI handles rendering
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
env.close()
