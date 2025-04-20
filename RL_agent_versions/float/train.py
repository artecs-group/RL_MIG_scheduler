import os
import csv
import argparse
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from env import SchedEnv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_steps", type=int, default=200000000, help="Num steps."
)
parser.add_argument(
    "--N", type=int, default=14, help="Max num ready tasks."
)
parser.add_argument(
    "--M", type=int, default=14, help="Discretization size."
)
parser.add_argument(
    "--device", type=str, default="cpu", help="Device for training (cpu or gpu)."
)
parser.add_argument(
    "--type_tasks", type=str, default="good_scaling", help="Type of tasks for training."
)
parser.add_argument(
    "--restore", type=str, default=None, help="Path to the model to restore for training."
)


if __name__ == "__main__":
    args = parser.parse_args()

    print("Type of tasks:", args.type_tasks)

    env = SchedEnv({"N": args.N, "M": args.M}, type_tasks = args.type_tasks) # Initialize env

    def mask_fn(env):
        # Call the method of the environment to get the action mask
        return env.valid_action_mask()


    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # Network architecture
    net_arch = dict(pi=[256, 256], vf=[256, 256])

    # PPO hyperparameters with action masking
    if args.restore:
        print("Loading pre-trained model...")
        model = MaskablePPO.load(args.restore, env=env, device=args.device)
        restored_num_timesteps = model.num_timesteps
    else:
        print("No pre-trained model found. Initializing a new model...")
        model = MaskablePPO(MaskableActorCriticPolicy, ent_coef=0.01, env=env, verbose=2, device=args.device, gamma=1)
        restored_num_timesteps = 0
    
    observation_space = env.observation_space
    action_space = env.action_space
    lr_schedule = lambda _: 0.0003
    model.policy = MaskableActorCriticPolicy(observation_space=observation_space, action_space=action_space, lr_schedule=lr_schedule, net_arch=net_arch)
    model.policy = model.policy.to(model.device)

    try:
        model.learn(args.num_steps)
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        model.policy_kwargs = {"net_arch": net_arch}  # Before saving the model, we need to set the policy_kwargs attribute to the net_arch
        # Save the model in a zip file
        model.save(f"./trained_models/bs3_N={args.N}_M={args.M}_s={restored_num_timesteps + args.num_steps}_{args.type_tasks}")