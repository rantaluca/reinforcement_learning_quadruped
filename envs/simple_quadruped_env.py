"""
@ANTALUCA Robert 2024-2025

This is an environment for a spot inspired quadruped robot to learn to follow a given twist command.
The robot has 12 joints and gets observations from:
    Joint positions, joint velocities, his linear and angular velocity (simulated imu).

Environment is based on the OpenAI gym, stable baselines3 for the algorithms and Pybullet physics engine.

Code formatted with black
Pylint score: 9.32/10
"""

import math
import os
from turtle import done
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym


class QuadrupedEnv(gym.Env):
    """
    A custom OpenAI Gym environment for a quadruped robot to learn to follow a given twist command.
    The robot has 12 joints and gets observations from joint positions, joint velocities, and his linear and angular velocity (IMU).
    The environment uses PyBullet as the physics engine.
    """

    def __init__(self, gui=True, ctrl_hz=10, sim_hz=500):
        """Initialize the QuadrupedEnv environment."""
        super(QuadrupedEnv, self).__init__()
        self.ctrl_hz = ctrl_hz
        self.sim_hz = sim_hz
        self.sim_substeps = int(sim_hz // ctrl_hz)
        self.gui = gui
        self.debug_reward_id = None

        self.kp = 1.2  # position gain
        self.kd = 0.2  # velocity gain
        self.tau = 4.5  # max torque

        # Config pybullet
        p.connect(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # pybullet models lib
        # p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Enable shadows
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)  # Disable wireframe
        p.setPhysicsEngineParameter(numSolverIterations=50)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        p.setPhysicsEngineParameter(erp=0.2, contactERP=0.05, frictionERP=0.2)
        p.setPhysicsEngineParameter(enableConeFriction=1, deterministicOverlappingPairs=1)
        p.setPhysicsEngineParameter(enableFileCaching=0)

        # roblot parameters
        self.robot_id = None
        self.actuated_ids = None
        self.q0 = None  # initial pose
        self.limits = None

        # Action space is the 12 joints of the robot
        self.action_scale = np.deg2rad(25)  # +/- 25 degrees
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Observation space is the joint positions, joint velocities, linear and angular velocity from the imu and twist target
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32
        )  # 12 (joint positions) + 12 (joint velocities) + 6 (IMU data) + 3 (twist target)

        # those range are used to normalize the observations (better for PPO)
        self.joint_positions_range = np.pi
        self.joint_velocities_range = 0.3  # rad/s
        self.imu_linear_velocity_range = np.array([2.0, 2.0, 2.0])  # m/s
        self.imu_angular_velocity_range = np.array([5.0, 5.0, 5.0])  # rad/s

        # target/rewards parameters
        self.target_twist = np.array(
            [0.05, 0.0, 0.0], dtype=np.float32
        )  # will make it dynamic later
        self.target_twist_range = np.array(
            [0.5, 0.5, 1.0], dtype=np.float32
        )  # max values for normalization

        self.time_on_ground = 0
        self.alive_bonus = 0.0
        self.prev_base_xy = np.zeros(2, dtype=np.float32)
        self.last_50_positions = []

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state and return the initial observation."""
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.sim_hz)
        p.setPhysicsEngineParameter(enableConeFriction=1)

        # Adding the ground
        plane_id = p.loadURDF("plane.urdf")

        p.changeDynamics(
            plane_id,
            -1,
            lateralFriction=1.0,
            spinningFriction=0.01,
            rollingFriction=0.01,
        )  # friction settings for the ground
        # ground

        # Spawning the robot
        urdf_path = os.path.join(
            os.path.dirname(__file__), "../ressources/urdfs/spot/spot_v2.urdf"
        )
        self.robot_id = p.loadURDF(
            urdf_path,
            [0, 0, 0.20],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
        )

        # discover actuated joints (because of the lidar/imu links in urdf that are not actuated)
        ids, q0, lim = [], [], []
        for j in range(p.getNumJoints(self.robot_id)):
            ji = p.getJointInfo(self.robot_id, j)
            if ji[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                ids.append(j)
                q0.append(0.0)
                lim.append([ji[8], ji[9]])
        self.actuated_ids = ids[:12]
        self.q0 = np.array(q0[:12], np.float32)
        self.limits = np.array(lim[:12], np.float32)

        # disable defaults and set neutral
        for j in range(p.getNumJoints(self.robot_id)):
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0.0)
            # p.setJointMotorControl2(self.robot_id, j, p.TORQUE_CONTROL, force=0.0)
        for k, j in enumerate(self.actuated_ids):
            p.resetJointState(self.robot_id, j, float(self.q0[k]))

        # reset proprio variables
        self.time_on_ground = 0
        self.prev_base_xy = np.array(
            p.getBasePositionAndOrientation(self.robot_id)[0][:2], np.float32
        )
        self.last_50_positions = [self.prev_base_xy.tolist()]
        self.alive_bonus = 0.0

        return self._get_observation(), {}

    def _follow_robot(self):
        """Make the camera follow the robot."""
        if not self.gui:
            return
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=pos
        )

    def _simulate_imu(self, noise=False):
        """Simulate the IMU data with some noise."""
        # Simulates the imu data
        base_velocity, base_angular_velocity = p.getBaseVelocity(self.robot_id)
        imu_data = np.asarray(
            list(base_velocity) + list(base_angular_velocity), dtype=np.float32
        )
        if noise:
            imu_data += self.np_random.normal(
                0, 0.02, imu_data.shape
            )  # noise for better sim2real robustness
        return imu_data

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

    def _get_observation(self):
        """Get the current observation from the environment.
        Returns:
            observation (np.array): normalized observation array of shape (33,).
        """

        # Getting the observation:
        joint_states = p.getJointStates(self.robot_id, self.actuated_ids)

        joint_positions = np.asarray(
            [js[0] for js in joint_states], dtype=np.float32
        )  # 12
        joint_velocities = np.asarray(
            [js[1] for js in joint_states], dtype=np.float32
        )  # 12

        imu_data = np.asarray(self._simulate_imu(noise=True), dtype=np.float32)  # 6

        target = np.asarray(self.target_twist, dtype=np.float32)  # 3

        # Normalizing the observation
        epsilon = 1e-8  # to avoid division by zero

        joint_positions_norm = joint_positions / (self.joint_positions_range + epsilon)
        joint_velocities_norm = joint_velocities / (
            self.joint_velocities_range + epsilon
        )
        imu_linear_velocity_norm = imu_data[:3] / (
            self.imu_linear_velocity_range + epsilon
        )
        imu_angular_velocity_norm = imu_data[3:] / (
            self.imu_angular_velocity_range + epsilon
        )
        target_norm = target / (self.target_twist_range + epsilon)

        observation = np.concatenate(
            [
                joint_positions_norm,
                joint_velocities_norm,
                imu_linear_velocity_norm,
                imu_angular_velocity_norm,
                target_norm,
            ]
        ).astype(np.float32)

        observation = np.clip(observation, -1.5, 1.5)  # clipping for safety

        # asserts
        assert observation.shape == (33,), f"obs shape {observation.shape}"
        if not np.all(np.isfinite(observation)):
            raise ValueError("Non-finite obs")

        return observation

    def compute_reward(self):
        """Compute the reward based on the current state of the robot."""

        # getting position of the robot
        base_position, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        rot_mat = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)

        # Uprightness penalty
        z_base = rot_mat[:, 2]
        uprightness = float(
            np.dot(z_base, np.array([0, 0, 1]))
        )  # cosine between z_base and the world z axis
        uprightness = np.clip(uprightness, -1.0, 1.0)  # for safety

        # Velocity penalty  
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_id)
        vel_error = np.linalg.norm(linear_velocity[:2] - self.target_twist[:2])
        ang_vel_error = abs(angular_velocity[2] - self.target_twist[2])

        # Going forward in the right direction reward
        base_xy = np.array(base_position[:2], dtype=np.float32)
        step_disp_xy = base_xy - self.prev_base_xy
        desired_dir = self.target_twist[:2] / (
            np.linalg.norm(self.target_twist[:2]) + 1e-6
        )  # unit vec
        progress_along_dir = float(step_disp_xy @ desired_dir)  # meters
        self.prev_base_xy = base_xy

        # alive bonus for not falling
        self.alive_bonus += 0.1


        # Weights
        w_vel = 5.3  # planar velocity tracking
        w_yaw = 3.3  # yaw 
        w_upright = 0.1  # uprightness
        w_prog = 5.1  # progress shaping
        w_vz = 0.1  # vertical slip penalty
        w_alive = 0.3  # alive bonus

        reward = (
            -w_vel * vel_error
            - w_yaw * ang_vel_error
            + w_upright * uprightness
            + w_prog * progress_along_dir
            - w_vz * abs(linear_velocity[2])
            + w_alive * self.alive_bonus
    )*10.0  # scaling for better rewards
        return reward

    def _termination(self):
        """
        Termination logic:
        - done  : (use for fall/flip elsewhere if needed)
        - trunc : (a) clear stagnation or (b) sustained misalignment w.r.t. target direction
        """
        done = False
        trunc = False
        term_reward = 0.0

        if len(self.last_50_positions)>140:
            # mean of the orientation of the last 50 positions
            for i in range(1, len(self.last_50_positions)):
                vec = np.array(self.last_50_positions[i]) - np.array(self.last_50_positions[i-1])
                if np.linalg.norm(vec)>1e-3:
                    vec /= np.linalg.norm(vec)
                    angle = math.atan2(vec[1], vec[0])
                    break
            mean_vec = np.array([math.cos(angle), math.sin(angle)])
            target_vec = self.target_twist[:2] / (np.linalg.norm(self.target_twist[:2])+1e-6)
            alignment = float(np.dot(mean_vec, target_vec))  # cosine between the two
            if alignment < 0.5:
                term_reward = -20.0
                trunc = True
                print("Terminated: misalignment")
        return done, trunc, term_reward

    def step(self, action):
        """Called at every step of the episode to apply the action given as a parameter.
        Returns the new observation, reward, done, and info."""

        action = np.asarray(action, np.float32)  # ensure action is a numpy array
        q_cmd = self.q0 + self.action_scale * action
        q_cmd = np.clip(q_cmd, self.limits[:, 0], self.limits[:, 1])

        for _ in range(self.sim_substeps):
            for k, j in enumerate(self.actuated_ids):
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=q_cmd[k],
                    positionGain=self.kp,
                    velocityGain=self.kd,
                    force=self.tau,
                )
            p.stepSimulation()

        obs = self._get_observation()
        reward = self.compute_reward()
        if self.gui:
            if self.debug_reward_id is not None:
                p.removeUserDebugItem(self.debug_reward_id)
            # reward + details
            self.debug_reward_id = p.addUserDebugText(
                f"Reward: {reward:.2f}",
                self.prev_base_xy.tolist() + [0.5],
                textColorRGB=[1, 0, 0] if reward < 0 else [0, 1, 0],
                textSize=2.5,
            ) 
        # update last 50 positions
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        current_xy = np.array(base_pos[:2], dtype=np.float32)
        self.last_50_positions.append(current_xy.tolist())
        if len(self.last_50_positions) > 250:
            self.last_50_positions.pop(0)

        done, trunc, term_reward = self._termination()
        self._follow_robot()
        return obs, reward+term_reward, done, trunc, {}

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()
