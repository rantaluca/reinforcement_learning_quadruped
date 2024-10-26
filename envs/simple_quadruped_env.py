#ANTALUCA Robert 2024

#This is a simple RL environment for my spot inspired quadruped robot to learn to reach a target position.
#The robot has 12 joints and gets information about its joint positions, joint velocities, is linear and angular velocity from a simulated imu and data from a simulated lidar.

#Environment is based on the OpenAI gym interface with stable baselines3 for the RL algorithms and uses the Pybullet physics engine.

import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO


class QuadrupedEnv(gym.Env):
    def __init__(self):
        super(QuadrupedEnv, self).__init__()
        
        #Action space is the 12 joints that can be controlled 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        #Observation space is the joint positions, joint velocities, linear and angular velocity from the imu and data from the lidar

        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)  #Observation: Example state + LIDAR + IMU
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(51,), dtype=np.float32)  #Observation: state + LIDAR + IMU + Pose Estimation + Pose Target

        self.time_on_ground = 0
        
        p.connect(p.GUI)  # or p.DIRECT non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #Pybullet models library
    
    def reset(self):
        # New goal choosen randomly
        self.target_position = [np.random.uniform(3.0, 7.0), np.random.uniform(3.0, 7.0), np.random.uniform(0.4, 0.7)]
        self.target_orientation = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]
        #Called at the start of an episode
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("your_quadruped.urdf", [0, 0, 0.5])
        self.time_on_ground = 0
        # Adding the lidar
        self.lidar_joint_index = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.5])
        #Used to reset the robot to a starting position if it spends too much time on the ground
        return self._get_observation(), {}
    
    def step(self, action):
        #Called at every step of the episode, with the action chosen by the agent given as input
        for joint_index in range(p.getNumJoints(self.robot_id)):
            p.setJointMotorControl2(self.robot_id, joint_index, p.POSITION_CONTROL, action[joint_index])
        p.stepSimulation()
        #The resulting observation, reward, done and info are computed and returned
        truncated = False
        done = False
        obs = self._get_observation()
        reward = self.compute_reward(p.getBasePositionAndOrientation(self.robot_id)[0], self.target_position)
        #If the robot reach the point ot falls the episode is terminated
        contact_points = p.getContactPoints(bodyA=self.robot_id, linkIndexA=-1)
        #if the robot is on the ground for too long, the episode is terminated
        if contact_points:
            self.time_on_ground += 1
            if self.time_on_ground > 150:
                done = True
                truncated = True
        #if the robot is close enough to the target (orientation and position), the episode is terminated
        if np.linalg.norm(np.array(p.getBasePositionAndOrientation(self.robot_id)[0]) - np.array(self.target_position)) < 0.1 and np.linalg.norm(np.array(p.getBasePositionAndOrientation(self.robot_id)[1]) - np.array(self.target_orientation)) < 0.1:
            done = True
        return obs, reward, done, truncated, {}
    


