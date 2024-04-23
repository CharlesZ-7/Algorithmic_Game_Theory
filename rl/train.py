
from env import MyEnv

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import numpy as np


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
for _ in range(5):
    algo.train()
print(algo.train())



# save results
# save_result = algo.save()
# path_to_checkpoint = save_result.checkpoint.path
# print(
#     "An Algorithm checkpoint has been created inside directory: "
#     f"'{path_to_checkpoint}'."
# )

# to load in results
# Algorithm.from_checkpoint(path_to_checkpoint)
# or 
# my_new_ppo.restore(save_result)

# evaluation on a single step...
# print(algo.compute_single_action(np.int64(1)))