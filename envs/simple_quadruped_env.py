#ANTALUCA Robert 2024

#This is a simple RL environment for my spot inspired quadruped robot to learn to reach a target position.
#The robot has 12 joints and gets information about its joint positions, joint velocities, is linear and angular velocity from a simulated imu and data from a simulated lidar.

#Environment is based on the OpenAI gym interface with stable baselines3 for the RL algorithms and uses the Pybullet physics engine.

import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time
from stable_baselines3 import PPO


class QuadrupedEnv(gym.Env):
    def __init__(self):
        super(QuadrupedEnv, self).__init__()
        
        #Action space is the 12 joints that can be controlled 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        #Observation space is the joint positions, joint velocities, linear and angular velocity from the imu and twist target
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)  #Observation: Example state + LIDAR + IMU
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(66,), dtype=np.float32)  #Observation: state + LIDAR + IMU + Pose Estimation + Pose Target
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)  #14?? (joint positions) + 14??? (joint velocities) + 6 (IMU data) + 3 (twist target)

        self.time_on_ground = 0
        self.treshold_target = 0.05
        
        p.connect(p.GUI)  # or p.GUI non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #Pybullet models library
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
    
        #Random target position and orientation (not used anymore)
        # self.target_position = [np.random.uniform(3.0, 20.0), np.random.uniform(3.0, 20.0), np.random.uniform(0.4, 0.7)]
        # self.target_orientation = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]

        #target is now a linear velocity in a random one direction and an angular velocity 
        # self.target_twist = {
        #     'linear_x': np.random.uniform(-0.3, 0.3),  # linear velocity in the x-direction
        #     'linear_y': np.random.uniform(-0.3, 0.3),  # linear velocity in the y-direction
        #     'angular': np.random.uniform(-0.5, 0.5)    # angular velocity (rotation)
        # }

        #simple forward target
        self.target_twist = {
            'linear_x': 0,  # linear velocity in the x-direction
            'linear_y': 0.2,  # linear velocity in the y-direction
            'angular': 0    # angular velocity (rotation)
        }

        #Called at the start of an episode
        p.resetSimulation()
        p.setGravity(0, 0,-9.81)
        plane_id = p.loadURDF("plane.urdf")
        #Setting the friction for the ground with the robot
        p.changeDynamics(plane_id, -1, lateralFriction=1.5)
        # #adding a visual target ( with a sphere)
        # self.target_id = p.loadURDF("sphere_small.urdf", self.target_position)
        #Adding the robot
        self.robot_id = p.loadURDF("/ressources/urdfs/spot/spot_v1.urdf", [0, 0, 0])
        #reset the position of the robot
        for joint_index in range(p.getNumJoints(self.robot_id)):
            p.resetJointState(self.robot_id, joint_index, targetValue=0)
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])
        self.time_on_ground = 0

        #Used to reset the robot to a starting position if it spends too much time on the ground
        return self._get_observation(), {}
    
    def step(self, action):
        #Called at every step of the episode, with the action chosen by the agent given as input
        for joint_index in range(12):  # Limit to 12 joints
            p.setJointMotorControl2(self.robot_id, joint_index, p.POSITION_CONTROL, action[joint_index])
        p.stepSimulation()
        #The resulting observation, reward, done and info are computed and returned
        truncated = False
        done = False
        obs = self._get_observation()
        reward = self.compute_reward()
        #If the robot reach the point ot falls the episode is terminated
        contact_points = p.getContactPoints(bodyA=self.robot_id, linkIndexA=-1)

        #If the robot is on the ground for too long, the episode is terminated
        if contact_points:
            self.time_on_ground += 1
            if self.time_on_ground > 450:
                done = True
                truncated = True   
                    
        # Checking if position and orientation are close enough to the target
        # if (np.linalg.norm(np.array(position) - np.array(self.target_position)) < self.treshold_target and
        #     np.linalg.norm(np.array(robot_orientation_euler) - np.array(self.target_orientation)) < self.treshold_target):
        #     done = True

        base_velocity, base_angular_velocity = p.getBaseVelocity(self.robot_id)
        #Checking if the target velocity is reached
        # Checking if the target velocity is reached
        return obs, reward, done, truncated, {}
    
    def _get_observation(self):
        #Returns the  the joint positions, joint velocities, linear and angular velocity from the imu and data from the lidar
        joint_states = p.getJointStates(self.robot_id, range(p.getNumJoints(self.robot_id)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        # position, orientation = p.getBasePositionAndOrientation(self.robot_id) #to be commented if no position in observatio
        # orientation_euler = p.getEulerFromQuaternion(orientation)
        #lidar_data = self._simulate_lidar(position, orientation)
        imu_data = self._simulate_imu()
        #return np.array(joint_positions + joint_velocities + lidar_data + imu_data + list(position) + list(orientation_euler) + self.target_position + self.target_orientation)
        #return np.array(joint_positions + joint_velocities + lidar_data + imu_data + self.target_position + self.target_orientation)
        return np.array(joint_positions + joint_velocities + imu_data +  [self.target_twist['linear_x'], self.target_twist['linear_y'], self.target_twist['angular']])
    

    # def _simulate_lidar(self, position, orientation):
    #     lidar_data = []
    #     lidar_height = 0.1
    #     lidar_position = [position[0], position[1], position[2]+lidar_height]
    #     for i in range(20):  # Collect 20 points
    #         angle = i * (2 * np.pi / 20)  # Divide 360° into 20 segments
    #         ray_from = lidar_position  
    #         ray_to = [4 * np.sin(angle), 4 * np.cos(angle), position[2]+lidar_height]
    #         result = p.rayTest(ray_from, ray_to)
    #         distance = result[0][2] if result[0][0] != -1 else 4.0  # Max distance if no hit
    #         lidar_data.append(distance)
    #     return lidar_data
    
    def _simulate_imu(self):
        #Simulates the imu data
        base_velocity, base_angular_velocity = p.getBaseVelocity(self.robot_id)
        imu_data = list(base_velocity) + list(base_angular_velocity)
        return imu_data
    
    def compute_reward(self):
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_orientation_euler = p.getEulerFromQuaternion(orientation)
        base_velocity, base_angular_velocity = p.getBaseVelocity(self.robot_id)

        linear_velocity = np.array(base_velocity)
        angular_velocity = np.array(base_angular_velocity)
        
        linear_error = np.linalg.norm(linear_velocity[:2] - np.array([self.target_twist['linear_x'], self.target_twist['linear_y']]))
        angular_error = np.abs(angular_velocity[2] - self.target_twist['angular'])

        roll_angle = robot_orientation_euler[0]
        roll_reward = abs(roll_angle - math.pi)

        reward = - linear_error - angular_error - self.time_on_ground*0.2 - roll_reward*0.2
        return reward


    # def compute_reward_position(self):
    #     position, orientation = p.getBasePositionAndOrientation(self.robot_id)
    #     robot_orientation_euler = p.getEulerFromQuaternion(orientation)
    #     linear_velocity, base_angular_velocity = p.getBaseVelocity(self.robot_id)
    #     # Computes the positional distance and orientation to the target
    #     position_distance = np.linalg.norm(np.array(position) - np.array(self.target_position))
    #     orientation_distance = np.linalg.norm(np.array(orientation) - np.array(self.target_orientation))   
    #     # Calculates the magnitude of velocity in the x direction and y direction ( not used anymore in the reward function)
    #     forward_plane_velocity = math.sqrt(linear_velocity[0] ** 2 + linear_velocity[1] ** 2)
    #     #Calculates the roll angle ( meaning the robot is flipped over)
    #     roll_angle = orientation[0]
    #     roll_reward = -abs(roll_angle - math.pi)
    #     # Calculates the reward with position, orientation and time on ground penalties
    #     if position_distance > 0.3:
    #         reward = -position_distance*3 -self.time_on_ground*0.2 + roll_reward + 1.5*linear_velocity[0] 
    #     else:
    #         reward = -position_distance*3 -orientation_distance*3 - self.time_on_ground*0.2 + roll_reward + 1.5* linear_velocity[0]
    #     return reward
    
    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    

    


