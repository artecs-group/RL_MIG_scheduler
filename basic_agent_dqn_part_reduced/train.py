import os
from pprint import pprint
import pandas as pd
from tqdm import tqdm
os.chdir("./basic_agent_dqn_part_reduced")
import argparse
import ray
from ray.tune.registry import register_env
from model import DQNActionMaskModel
from env import SchedEnv
from ray.rllib.algorithms.dqn.dqn import DQNConfig

parser = argparse.ArgumentParser()
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
    ray.init()
    register_env("mig_scheduler", lambda _: SchedEnv({"N": args.N, "M": args.M}, type_tasks = args.type_tasks))

    train_batch_size = 256

    config = (
            DQNConfig()
            .environment('mig_scheduler', disable_env_checking=True)
            .rollouts(num_rollout_workers=4, 
                    num_envs_per_worker=1)
            .framework('torch')
            .training(
                    hiddens=[],
                    dueling=False,
                    model={"custom_model": DQNActionMaskModel},
                    train_batch_size=train_batch_size,
                    training_intensity=False,
                    gamma=1,
                    )
            .api_stack(enable_rl_module_and_learner=False)
            .evaluation(evaluation_num_workers=1, evaluation_interval=1000)
            .resources(num_gpus=0, 
                    num_cpus_per_worker=1, 
                    num_gpus_per_worker=0)  
            .debugging(log_level='ERROR') 
            .reporting(min_sample_timesteps_per_iteration=500)
            )

    algo = config.build()
    policy = algo.get_policy()
    print(policy.model)



    train_dict = {}
train_dict['episode'] = []
train_dict['episode_reward'] = []
train_dict['step_num'] = []

episode_num = 0
iteration_num = 1000
for iteration in tqdm(range(iteration_num)):
    result = algo.train()

    # pprint(result)
    # input("Espera")

    mean_step_num = result["env_runners"]['episode_len_mean']
    episode_num += result["env_runners"]["episodes_this_iter"]
    mean_reward = result["env_runners"]['episode_reward_mean']

    train_dict['episode'].append(episode_num)
    train_dict['episode_reward'].append(mean_reward)
    train_dict['step_num'].append(mean_step_num)
    print('\n')
    print('episode={}, step_num={:.2f}, mean_reward={:.2f}'.format(episode_num, mean_step_num, mean_reward))

    if (iteration+1) % 100 == 0:
        save_result = algo.save("./DQN_M{}_N{}_save_episode_{}".format(args.M, args.N, episode_num))
        path_to_checkpoint = save_result.checkpoint.path
        print(
            "An Algorithm checkpoint has been created inside directory: "
            f"'{path_to_checkpoint}'."
        )
        train_df = pd.DataFrame.from_dict(train_dict)
        train_df.to_csv('./DQN_ray_train_df_episode_{}'.format(episode_num), index=False)
        print('------ train_df is saved at episode={}'.format(episode_num))