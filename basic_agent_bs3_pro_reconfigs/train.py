import csv
import gymnasium as gym
import numpy as np
import argparse
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import EveryNTimesteps
import os
os.chdir("./basic_agent_bs3_pro_reconfigs")
from env import SchedEnv
from callbacks import CustomCallback


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_steps", type=int, default=100000000, help="Num steps."
)
parser.add_argument(
    "--N", type=int, default=15, help="Max num ready tasks."
)
parser.add_argument(
    "--M", type=int, default=35, help="Discretization size."
)

parser.add_argument(
    "--type-tasks", type=str, default="good_scaling", help="Type of tasks for training."
)


if __name__ == "__main__":
    args = parser.parse_args()

    print("Type of tasks:", args.type_tasks)

    env = SchedEnv({"N": args.N, "M": args.M}, type_tasks = args.type_tasks) # Initialize env
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    observation_space = env.observation_space
    action_space= env.action_space
    lr_schedule = lambda _: 0.0003
    net_arch = dict(pi=[512, 512], vf=[512, 512])
    model = MaskablePPO(MaskableActorCriticPolicy, ent_coef=0.01, env = env, verbose=2, device="cpu", gamma = 1)
    model.policy = MaskableActorCriticPolicy(observation_space=observation_space, action_space=action_space, lr_schedule=lr_schedule, net_arch=net_arch)
    model.policy = model.policy.to(model.device)
    my_callback = CustomCallback(M = args.M, N = args.N, type_tasks=args.type_tasks)

    with open(f'ratios_N{args.N}_M{args.M}_{args.type_tasks}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["step", "ratio makespan"])
    my_callback.calculate_ratio(args.N, args.M, model)

    try:
        periodic_callback = EveryNTimesteps(n_steps=100000, callback=my_callback)
        # Create or open the CSV file in write mode
        model.learn(args.num_steps, callback=periodic_callback)
    
    except KeyboardInterrupt:
        print("Training interrupted")

    model.policy_kwargs = {"net_arch": net_arch}
    model.save(f"./trained_models/bs3_N={args.N}_M={args.M}_s={model.num_timesteps}_{args.type_tasks}")


    # # Note that use of masks is manual and optional outside of learning,
    # # so masking can be "removed" at testing time
    # model.predict(observation, action_masks=valid_action_array)