#ANTALUCA Robert 2024

#This is a simple RL environment for my spot inspired quadruped robot to learn to reach a target position.
#The robot has 12 joints and gets information about its joint positions, joint velocities, is linear and angular velocity from a simulated imu and data from a simulated lidar.

#Environment is based on the OpenAI gym interface with stable baselines3 for the RL algorithms and uses the Pybullet physics engine.

import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from stable_baselines3 import PPO


class QuadrupedEnvMJ(gym.Env):
    def __init__(self):
        super(QuadrupedEnvMJ, self).__init__()

        # Load the MuJoCo model (adjust to the path of your URDF/ XML model)
        self.model = mujoco.MjModel.from_xml_path("../ressources/urdfs/spot/spot_v1.urdf")
        self.data = mujoco.MjData(self.model)

        # Action space is the 12 joints that can be controlled
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Observation space includes joint positions, velocities, IMU data, and LIDAR data
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(66,), dtype=np.float32)

        self.time_on_ground = 0
        self.threshold_target = 0.1

        # Initialize MuJoCo visualizer
        mujoco.mjv_defaultOption(self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Random target position and orientation
        self.target_position = [np.random.uniform(3.0, 20.0), np.random.uniform(3.0, 20.0), np.random.uniform(0.4, 0.7)]
        self.target_orientation = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Reset robot position and joints to starting state
        for joint_index in range(self.model.nq):
            self.data.qpos[joint_index] = 0  # Reset joint position
        self.data.qvel[:] = 0  # Reset joint velocities

        self.time_on_ground = 0
        mujoco.mj_forward(self.model, self.data)  # Apply physics step after reset
        return self._get_observation(), {}

    def step(self, action):
        # Apply action to the robot
        self.data.ctrl[:12] = action  # Set control inputs for the joints
        mujoco.mj_step(self.model, self.data)

        # Get observation, reward, and done status
        obs = self._get_observation()
        position = self.data.qpos[:3]
        orientation = self.data.qpos[3:6]  # Assuming orientation as quaternion or Euler angles
        linear_velocity = self.data.qvel[:3]

        done = False
        truncated = False
        reward = self.compute_reward(position, self.target_position, orientation, self.target_orientation, linear_velocity)

        # Check if the robot has fallen or reached the target
        if self.data.ncon > 0:  # Check contact points for ground contact
            self.time_on_ground += 1
            if self.time_on_ground > 450:
                done = True
                truncated = True

        # Check if close enough to target position and orientation
        if (np.linalg.norm(np.array(position) - np.array(self.target_position)) < self.threshold_target and
            np.linalg.norm(np.array(orientation) - np.array(self.target_orientation)) < self.threshold_target):
            done = True

        return obs, reward, done, truncated, {}

    def _get_observation(self):
        # Gather joint positions, velocities, IMU, LIDAR data, and target information
        joint_positions = self.data.qpos[:12].tolist()  # First 12 positions
        joint_velocities = self.data.qvel[:12].tolist()  # First 12 velocities
        position = self.data.qpos[:3]  # Position of the robot base
        orientation = self.data.qpos[3:6]  # Orientation of the robot base
        lidar_data = self._simulate_lidar(position, orientation)
        imu_data = self._simulate_imu()
        return np.array(joint_positions + joint_velocities + lidar_data + imu_data + list(position) + list(orientation) + self.target_position + self.target_orientation)

    def _simulate_lidar(self, position, orientation):
        # Simulate LIDAR data by casting rays in MuJoCo
        lidar_data = []
        lidar_height = 0.1
        lidar_position = [position[0], position[1], position[2] + lidar_height]
        for i in range(20):  # Collect 20 points
            angle = i * (2 * np.pi / 20)
            ray_to = [lidar_position[0] + 4 * np.sin(angle), lidar_position[1] + 4 * np.cos(angle), lidar_position[2]]
            # Approximate ray casting (MuJoCo doesn’t support rayTest like PyBullet)
            distance = self._ray_test(lidar_position, ray_to)
            lidar_data.append(distance)
        return lidar_data

    def _ray_test(self, ray_from, ray_to):
        # Simulate ray casting in MuJoCo using distance approximation
        # Placeholder as MuJoCo lacks built-in ray casting
        return 4.0  # Default max distance if no hit is detected

    def _simulate_imu(self):
        # Get linear and angular velocities as IMU data
        base_velocity = self.data.qvel[:3]
        base_angular_velocity = self.data.qvel[3:6]
        imu_data = list(base_velocity) + list(base_angular_velocity)
        return imu_data

    def compute_reward(self, position, target_position, orientation, target_orientation, linear_velocity):
        # Compute distance to target and other factors
        position_distance = np.linalg.norm(np.array(position) - np.array(target_position))
        orientation_distance = np.linalg.norm(np.array(orientation) - np.array(target_orientation))

        # Velocity in x-y plane
        forward_plane_velocity = math.sqrt(linear_velocity[0] ** 2 + linear_velocity[1] ** 2)

        # Calculate roll angle penalty
        roll_angle = orientation[0]  # Roll angle approximation
        roll_reward = -abs(roll_angle - math.pi)

        # Total reward with position, orientation, and penalties
        if position_distance > 0.3:
            reward = -position_distance - self.time_on_ground + roll_reward + 5 * forward_plane_velocity
        else:
            reward = -position_distance - orientation_distance - self.time_on_ground + roll_reward + 5 * forward_plane_velocity
        return reward

    def render(self, mode='human'):
        mujoco.mjv_updateScene(self.model, self.data)

    def close(self):
        mujoco.mj_close(self.model)
    

    


