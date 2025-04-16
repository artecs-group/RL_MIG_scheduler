from sb3_contrib.ppo_mask import MaskablePPO
import argparse
import os
import numpy as np
os.chdir("./one_hot")
from env import SchedEnv
import re
from render import Window
from pprint import pprint
from utils import _action_to_str

parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename", type=str, help="Filename with the model to evaluate."
)


def evaluate(model, env, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs, _ = env.reset()
  for i in range(num_steps):
    # _states are only useful when using LSTM policies
    action, _states = model.predict(obs, action_masks=env.valid_action_mask())
    obs, reward, terminated, _, _ = env.step(action)
    print(_action_to_str(action))
    print("Reward", reward)
    print()
    print("Partition", env.obs["partition"])
    print("Slices_t", env.obs["slices_t"])
    print("Ready", env.obs["ready_tasks"])
    print()
    # Stats
    episode_rewards[-1] += reward
    if terminated:
        print("Reward", episode_rewards[-1])
        input("Espera")
        obs, _ = env.reset()
        episode_rewards.append(0.0)
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
  return mean_100ep_reward

if __name__ == "__main__":
    args = parser.parse_args()

    pattern = r"N=(\d+)_M=(\d+)_s=(\d+)"
    match = re.search(pattern, args.filename)

    N = int(match.group(1))  # Primer grupo es N
    M = int(match.group(2))  # Segundo grupo es M
    steps = int(match.group(3))  # Tercer grupo es s

    model = MaskablePPO.load(f"./trained_models/{args.filename}")

    env = SchedEnv({"N": N, "M": M})
    # mean_reward = evaluate(model, env, num_steps=10000)
    initial_obs, _ = env.reset()
    window = Window(env, model_trained=model)
