import argparse
import os

import gymnasium
import highway_env
import torch

from zeus.monitor import ZeusMonitor

from envs.MultiHighway import MultiHighway, multi_highway_env_creator
from models.BaselineTorchModel import BaselineTorchModel
from metrics.rewards import compute_influence_reward
from agentic.BaselineAgent import BaselineAgent
from agentic.algorithms import MAPPO

from utils.default_args import add_default_args

import ray
from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

parser = argparse.ArgumentParser()
add_default_args(parser)


# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print("using ", device)


def ray_routine(num_episodes,config):
    """THIS METHOD HAS BEEN DEPRECATED BECAUSE OF RLLIB NOT WORKING PROPERLY"""
    ModelCatalog.register_custom_model("baseline_model",BaselineTorchModel)
    register_env("multi_highway",multi_highway_env_creator)

    ray.init()
    ppo_conf=(ppo.PPOConfig()
        .api_stack( enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)
        .environment("multi_highway",env_config={"num_agents":5})
        .framework("torch")
        .training(
            model={
                "custom_model":"baseline_model",
                "custom_model_config":{}
            },
            optimizer={"type":torch.optim.Adam,"learning_rate":config.lr}
            )
        )
    mappo=ppo_conf.build()
    mappo.train()

    


def routine(num_episodes,config):
    if config.model=="baseline":
        agent_type=BaselineAgent
        model_type=BaselineTorchModel
                
    env=MultiHighway(config.num_agents)
    obs_space=env.env.observation_space
    #print(obs_space," obs_space\n\n")
    model_action_space=env.env.action_space[0]
    #print(model_action_space," model_act_space\n\n")
    num_outputs=2
    agents=[]
    for i in range(config.num_agents):
        agents.append(agent_type(f"agent_{i}",model_type,obs_space,model_action_space,num_outputs,device))

    if config.load:
        for agent,model_file in zip(agents,os.listdir(config.load_dir)):
            checkpoint=torch.load(f'{config.load_dir}/{model_file}',map_location=device)
            agent.model.load_state_dict(checkpoint['agent_policy_net_state_dict'])

    mappo=MAPPO(env.env,device,config)
    mappo.train(num_episodes,config.rollout_max_steps,agents)
    

    

        
if __name__ == "__main__":
    parsed_args = parser.parse_args()
    routine(parsed_args.num_episodes,parsed_args)