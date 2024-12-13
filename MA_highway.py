import argparse

import gymnasium
import highway_env
import torch

from zeus.monitor import ZeusMonitor

from envs.MultiHighway import MultiHighway
from models.BaselineTorchModel import BaselineTorchModel
from metrics.rewards import compute_influence_reward
from agentic.BaselineAgent import BaselineAgent

from utils.default_args import add_default_args

parser = argparse.ArgumentParser()
add_default_args(parser)


# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print("using ", device)


def routine(num_episodes,config):
    if config.model=="baseline":
        agent_type=BaselineAgent
        model_type=BaselineTorchModel
        
    env=MultiHighway(config.num_agents)
    obs_space=env.env.observation_space
    print(obs_space," obs_space\n\n")
    model_action_space=env.env.action_space[0]
    print(model_action_space," model_act_space\n\n")
    num_outputs=None
    agents=[]
    for i in range(config.num_agents):
        agents.append(agent_type(model_type,f"agent_{i}",obs_space,model_action_space,num_outputs,device))

    for ep in range(num_episodes):
        env.reset(seed=0)
        if config.render_env is True:
            env.render()

        
if __name__ == "__main__":
    parsed_args = parser.parse_args()
    routine(1,parsed_args)