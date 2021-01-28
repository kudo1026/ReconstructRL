import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import config
from env import EnvTruckDiscrete

if __name__ == "__main__":

    env_truck = EnvTruckDiscrete(verbose=config.verbose, vis=config.vis)

    model = PPO('MlpPolicy', env_truck, verbose=2)

    # before training evaluation
    mean_reward, std_reward = evaluate_policy(model, env_truck, n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # training
    model.learn(total_timesteps=1)

    # after training evaluation
    mean_reward, std_reward = evaluate_policy(model, env_truck, n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


