import os
import gymnasium as gym
from stable_baselines3 import PPO
#from envs.simple_quadruped_env_mj import QuadrupedEnvMJ
from envs.simple_quadruped_env import QuadrupedEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# def a main with argparser to choose gui or not: 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a quadruped robot using PPO.")
    parser.add_argument("--gui", action="store_true", help="Enable GUI for the environment.")
    parser.add_argument("--timesteps", type=int, default=1000000000, help="Total timesteps for training.")
    parser.add_argument("--save-freq", type=int, default=10000, help="Frequency of saving the model.")
    args = parser.parse_args()

    # Creating the environment with or without GUI
    if args.gui:
        use_gui = True
    else:
        use_gui = False

    # Model path
    model_save_path = "models/ppo_quadruped"
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path='./models/', name_prefix='ppo_quadruped_checkpoint')

    mk_env = lambda: QuadrupedEnv(gui=use_gui)
    venv = DummyVecEnv([mk_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=5.0)
    print("Environment created.")

    model = PPO(
    "MlpPolicy", venv,
    n_steps=2048, batch_size=256,
    gamma=0.99, gae_lambda=0.95,
    ent_coef=0.005, learning_rate=3e-4,
    clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5,
    verbose=1)
    print("Created a new model.")

    try:
        # Training the model 
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model.")
        model.save(model_save_path)

    model.save(model_save_path)
    print("Model saved at", model_save_path)

    # Closing the env 
    env.close()




