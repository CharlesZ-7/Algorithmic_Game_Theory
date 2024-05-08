
from env import MyEnv

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import numpy as np
import sys, os


# add custom env to registry
def env_creator(env_config):
    return MyEnv(env_config)
register_env("my_env", env_creator)

# initialize training
ray.init()
config = (
    PPOConfig()
    .environment(env="my_env", env_config={})
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [256, 256]}, lr=0.001)
    .evaluation(evaluation_num_workers=1)
)
algo = config.build()

# training
metrics_to_print = [
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
    # "episode_len_mean",
]

max_reward = 0

for _ in range(100):
    results = algo.train()
    print({k: v for k, v in results.items() if k in metrics_to_print})

    # save highest checkpoint
    if results["episode_reward_mean"] > max_reward:
        max_reward = results["episode_reward_mean"]

        # save results
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        save_result = algo.save(checkpoint_dir=checkpoint_dir)
        path_to_checkpoint = save_result.checkpoint.path
        print(
            "An Algorithm checkpoint has been created inside directory: "
            f"'{path_to_checkpoint}'."
        )

# to load in results
# Algorithm.from_checkpoint(path_to_checkpoint)
# or 
# my_new_ppo.restore(save_result)

# evaluation on a single step...
# print(algo.compute_single_action(np.int64(1)))



# NOTE:
    # the actual training isn't quite working...
    # make the problem easier or something or just different in general...

    # write up some code to actually load in the model and test it against some other agents to see progress...