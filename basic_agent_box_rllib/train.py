import os
os.chdir("./basic_agent_box_rllib")
import argparse
import ray
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from model import TorchParametricActionsModel
from env import SchedEnv
from callbacks import CustomMetricsCallback
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.utils.test_utils import check_learning_achieved
# from ray.rllib.utils.metrics import (
#     ENV_RUNNER_RESULTS,
#     EPISODE_RETURN_MEAN,
#     NUM_ENV_STEPS_SAMPLED_LIFETIME,
# )

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=200, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=150.0, help="Reward at which we stop training."
)
parser.add_argument(
    "--N", type=int, default=15, help="Max num ready tasks."
)
parser.add_argument(
    "--M", type=int, default=7, help="Discretization size."
)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    register_env("mig_scheduler", lambda _: SchedEnv({"N": args.N, "M": args.M}))
    ModelCatalog.register_custom_model(
        "action_mask_model",
        TorchParametricActionsModel
    )

    cfg = {}

    
    config = dict(
        {
            "env": "mig_scheduler",
            "model": {
                "custom_model": "action_mask_model",
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            "num_env_runners": 2,
            "framework": "torch",
            "callbacks": CustomMetricsCallback,
            "evaluation_parallel_to_training": True,
            "evaluation_num_env_runners" : 1,
            "evaluation_interval": 1,
            "evaluation_duration": "auto",
            "evaluation_duration_unit": "timesteps",
            "evaluation_config": {
                "explore": False,
                "metrics_num_episodes_for_smoothing": 5,          
            },
            "rollout_fragment_length": 200,  # Aumenta este valor
            "sgd_minibatch_size": 128,  # Aumenta este valor
            "train_batch_size": 4000,  # Aumenta este valor
        },
        **cfg,
    )

    stop = {
        #TRAINING_ITERATION: args.stop_iters,
        TRAINING_ITERATION: 1,
        f"{NUM_ENV_STEPS_SAMPLED_LIFETIME}": args.stop_timesteps,
        #f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
    }

    tuner = tune.Tuner(
        args.run,
        run_config=air.RunConfig(stop=stop, verbose=2),
        param_space=config,
    )
    
    results = tuner.fit()

    print(results)


    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()