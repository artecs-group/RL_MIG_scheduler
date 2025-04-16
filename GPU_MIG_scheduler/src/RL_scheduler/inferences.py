from sb3_contrib.ppo_mask import MaskablePPO
import argparse
import os
import numpy as np
from env import SchedEnv
import re
from pprint import pprint
from utils import action_to_str, makespan_lower_bound
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", type=str, help="Filename with the model to evaluate."
)
parser.add_argument(
    "--task_path", type=str, help="Filename with the task info for scheduling."
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
    print(action_to_str(action))
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


def load_tasks(task_path):
    """
    Load the tasks from a file.
    :param task_path: (str) path to the file with the tasks
    :return: (list) list of tasks
    """
    task_info = []
    with open(task_path, "r") as f:
        for line in f:
            if line.startswith("Task"):
                task_info.append([line.strip()])
            else:
                instance_size, time = line.split()
                time = float(time)
                task_info[-1].append(time)            
    return task_info


def create_env(model_path):
    pattern = r"N=(\d+)_M=(\d+)_s=(\d+)_(.*).zip"
    match = re.search(pattern, model_path)

    N = int(match.group(1))  # Primer grupo es N
    M = int(match.group(2))  # Segundo grupo es M
    steps = int(match.group(3))  # Tercer grupo es s
    type_tasks = match.group(4)  # Cuarto grupo es type_tasks
    print(type_tasks)

    env = SchedEnv({"N": N, "M": M}, type_tasks=type_tasks)
    return env

def compute_actions(env, tasks_info):
    options = {"ready_tasks": tasks_info}
    obs, _ = env.reset(options=options)
    print("Initial observation:", obs)
    model = MaskablePPO.load(args.model_path)
    done = False
    while not done:
        action, _ = model.predict(obs, action_masks=env.valid_action_mask())
        obs, reward, done, _, _ = env.step(action)
    print(env.actions)
    return env.actions

def write_actions_to_file(actions, output_path):
    """
    Write the actions to a file.
    :param actions: (list) list of actions
    :param output_path: (str) path to the file to write the actions
    """
    with open(output_path, "w") as f:
        for action_name, action_options in actions:
            output = f"{action_name} "
            for option in action_options:
                output += f"{option} "
            f.write(f"{output.strip()}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    
    env = create_env(args.model_path)
    tasks_info = load_tasks(args.task_path)
    print("Tasks info:", tasks_info)
    actions = compute_actions(env, tasks_info)
    write_actions_to_file(actions, "./tmp/schedule.txt")
    
