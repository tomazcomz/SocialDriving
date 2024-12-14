import torch
from torch import multiprocessing,nn

import gymnasium as gym

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (RewardSum,Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from agentic.Agent import Agent

from collections import defaultdict, namedtuple, deque
from tqdm import tqdm
import random

import warnings
warnings.filterwarnings("ignore")

#replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

#save transitions to memory
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class MAPPO():

    def __init__(self,input_env:gym.Env,device:torch.device,model:nn.Module,config):

        self.device=device
        self.model=model
        self.config=config

        # When calling MAPPO, we should pass input_env=MultiHighway.env
        self.env=input_env

    
    def rollout(env:gym.Env,max_steps:int,agents:list[Agent],device:torch.device)->ReplayMemory:
        states, info = env.reset()
        agent_memories = {i : ReplayMemory(10000) for i in range(len(states))}
        terminated=False
        actions_tuple = tuple()
        episode_reward=0
        while not terminated:
            for i_state in range(len(states)):
                actions_tuple = actions_tuple + (agents[i_state].action(), )
            observation, reward, terminated, truncated, info = env.step(actions_tuple)
            episode_reward+=reward
            done = terminated or truncated
            next_states = observation
            for i_agent in range(len(states)):
                state = states[i_agent].flatten()
                if not done:
                    next_state = torch.tensor([next_states[i_agent].flatten()], device=device)
                else:
                    next_state = None
                agent_memories[i_agent].push(torch.tensor([state], device=device), torch.tensor([actions_tuple[i_agent]], device=device), next_state, torch.tensor([reward], device=device))        
            states = next_states
        return agent_memories


    

    def train(self,num_episodes:int,max_steps:int,agents:list[Agent]):
        pbar = tqdm(total=num_episodes, desc="episode_reward_mean = 0")
        episode_reward_mean_list = []
        for ep in range(num_episodes):
            memories=self.rollout(self.env,max_steps,agents,self.device)
            for replay,agent in zip(memories,agents):
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*replay.memory))

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = agent.model.forward(state_batch).gather(1, action_batch.unsqueeze(0))

                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1).values
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(len(replay.memory), device=self.device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = agent.model.value_fucntion(non_final_next_states).max(1).values
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch

                advantage=state_action_values-expected_state_action_values