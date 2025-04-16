import gymnasium as gym
import numpy as np
import argparse
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import os
os.chdir("./one_hot")
from env import SchedEnv


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
    "--N", type=int, default=14, help="Max num ready tasks."
)
parser.add_argument(
    "--M", type=int, default=7, help="Discretization size."
)
parser.add_argument(
    "--type_task", type=str, default="good_scaling", help="Type of workload."
)

if __name__ == "__main__":
    args = parser.parse_args()

    env = SchedEnv({"N": args.N, "M": args.M}, type_tasks=args.type_task) # Initialize env
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=2, device="cpu", gamma = 1)

    try:
        model.learn(args.num_steps)
    except KeyboardInterrupt:
        print("Training interrupted")
        model.save(f"./trained_models/bs3_N={args.N}_M={args.M}_t={args.type_task}_s={model.num_timesteps}")

    model.save(f"./trained_models/bs3_N={args.N}_M={args.M}_t={args.type_task}_s={model.num_timesteps}")

    # # Note that use of masks is manual and optional outside of learning,
    # # so masking can be "removed" at testing time
    # model.predict(observation, action_masks=valid_action_array)