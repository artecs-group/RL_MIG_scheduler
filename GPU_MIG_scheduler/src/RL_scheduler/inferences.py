from sb3_contrib.ppo_mask import MaskablePPO
import argparse
from env import SchedEnv
import re
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", type=str, help="Filename with the model to evaluate."
)
parser.add_argument(
    "--task_path", type=str, help="Filename with the task info for scheduling."
)
parser.add_argument(
    "--output_path", type=str, help="Filename with the path to write the schedule."
)

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
                _, name = line.split()
                name = name.strip()
                task_info.append([name])
            else:
                _, time = line.split()
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

    env = SchedEnv({"N": N, "M": M}, type_tasks=type_tasks)
    return env

def compute_actions(env, tasks_info):
    options = {"ready_tasks": tasks_info}
    obs, _ = env.reset(options=options)
    model = MaskablePPO.load(args.model_path)
    done = False
    while not done:
        action, _ = model.predict(obs, action_masks=env.valid_action_mask())
        obs, reward, done, _, _ = env.step(action)
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
    actions = compute_actions(env, tasks_info)
    write_actions_to_file(actions, args.output_path)
    
