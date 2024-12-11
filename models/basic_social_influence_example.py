#itializing environment

# -- REMINDER -- 
# Check Replay Memory, maybe delete after optimization step and get new batch?

import gymnasium
import pprint
import highway_env
from matplotlib import pyplot as plt
import numpy as np
import pickle
import warnings
from datetime import date
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from glob import glob


env = gymnasium.make('intersection-v1', render_mode=None)

#config
n_agents = 4
discrete = True

writer = SummaryWriter()

# Set action type based on discrete flag
action_type = "DiscreteMetaAction" if discrete else "ContinuousAction"


model_name = "only_agents"
# Multi-agent environment configuration
env.unwrapped.config.update({
"controlled_vehicles": n_agents,
"initial_vehicle_count": 0,
"observation": {
    "vehicles_count": n_agents,  
    "type": "MultiAgentObservation",
    "observation_config": {
    "type": "Kinematics",
    }
},
"action": {
    "type": "MultiAgentAction",
    "action_config": {
    "type": action_type,
    "lateral": False,
    "longitudinal": True
    }
}
})
env.reset()


#print config of env
print("configuration dict of environment:")
pprint.pprint(env.unwrapped.config)

#learning
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print("using ", device)

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

#Q-network
class PolicyNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

#hyperparameters and utilities
BATCH_SIZE = 10
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-5

#Get number of actions from gym action space
n_actions = 3
# Get the number of state observations
n_observations = 25
render_env = False
agents = {i: PolicyNetwork(n_observations, n_actions) for i in range(n_agents)}
optimizers = {i: optim.Adam(agents[i].parameters(), lr=1e-3) for i in range(n_agents)}
agent_memories = {i : ReplayMemory(10000) for i in range(n_agents)}


# in case you want to add arguments:
# if args.load_model:
#     if args.model_path is None:
#         raise ValueError("Model path must be specified when load_model is True")
    
#     checkpoint = torch.load(args.model_path)
#     policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
#     target_net.load_state_dict(checkpoint['target_net_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     episode_rewards = checkpoint['episode_rewards']
#     episode_durations = checkpoint['episode_durations']
#     starting_episode = checkpoint['episode'] + 1
#     print(f"Loaded model from {args.model_path}, continuing from episode {starting_episode}")
# else:
#     target_net.load_state_dict(policy_net.state_dict())
#     starting_episode = 0

steps_done = 0


def select_action(state, policy_net):
    policy_net.to(device)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample < eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state = state.flatten()
            state = torch.tensor(state, device=device)
            return True, torch.argmax(policy_net(state)).item()
    else:
        return False, env.action_space.sample()

episode_durations = []
episode_rewards = []


def optimize_model(policy_net, optimizer, memory):
    target_net = policy_net
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(0))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def compute_influence_reward(state, next_state, actions):
    """Compute the influence reward for each agent."""
    influence_rewards = {}
    for i in range(n_agents):
        other_agents = [j for j in range(n_agents) if j != i]
        original_action_dists = {
            j: torch.softmax(agents[j](state[j]), dim=-1) for j in other_agents
        }
        counterfactual_dists = {j: [] for j in other_agents}

        for counterfactual_action in range(n_actions):
            for j in other_agents:
                counterfactual_logits = agents[j](state[j]).clone()
                counterfactual_logits[0, counterfactual_action] += 1e-6
                counterfactual_dists[j].append(torch.softmax(counterfactual_logits, dim=-1))

        total_influence = 0.0
        for j in other_agents:
            counterfactual_mean_dist = torch.stack(counterfactual_dists[j]).mean(0)
            kl_div = torch.sum(
                original_action_dists[j] * torch.log(original_action_dists[j] / (counterfactual_mean_dist + 1e-9))
            )
            total_influence += kl_div.item()

        influence_rewards[i] = total_influence
    return influence_rewards

starting_episode = 0
num_episodes = 100000

for i_episode in range(starting_episode, num_episodes):
    # Initialize the environment and get its state
    states, info = env.reset()
    print(states[0].flatten().shape)
    print(states[0])
    print(f"running episode {i_episode}: ")

    episode_reward = 0

    for t in count():

        actions_tuple = tuple()

        for i_state in range(len(states)):
            print(states[i_state])
            policy_net_action, action = select_action(states[i_state], agents[i_state])
            if not policy_net_action:
                actions_tuple = action
                break
            else:
                actions_tuple = actions_tuple + (action, )
        
        print(actions_tuple)

        observation, reward, terminated, truncated, info = env.step(actions_tuple)
        #print("got reward ", reward)
        episode_reward+=reward
        reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
        done = terminated or truncated
        if render_env is True:
            env.render()

        if terminated:
            next_states = None
        else:
            next_states = observation
        
        for i_agent in range(n_agents):
            state = states[i_agent].flatten()
            if next_states is not None:
                next_state = torch.tensor([next_states[i_agent].flatten()], device=device)
            else:
                next_state = None
            agent_memories[i_agent].push(torch.tensor([state], device=device), torch.tensor([actions_tuple[i_agent]], device=device), next_state, torch.tensor([reward_t], device=device))        

        # Move to the next state
        states = next_states

        # Perform one step of the optimization (on the policy network)
        for i_agent in agents:
            loss = optimize_model(agents[i_agent], optimizers[i_agent], agent_memories[i_agent])

        # Log metrics to tensorboard
        writer.add_scalar('Training/Episode Duration', t + 1, i_episode)
        writer.add_scalar('Training/Episode Reward', reward, i_episode)
        if loss is not None:
            writer.add_scalar('Training/Loss', loss.item(), i_episode)

        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(episode_reward)
            # plot_durations()
            # plot_rewards()
            print(f"episode finished with {t+1} steps")
            break
        

        # # Save models and data every 1000 episodes
        # if (i_episode + 1) % 500 == 0:
        #     # Save model parameters using PyTorch
        #     model_save_path = f"saved_models/"+"model_"+model_name+date.today().strftime('%Y-%m-%d')+"_episode_"+str(i_episode)+".pt"
        #     torch.save({
        #         'episode': i_episode,
        #         'policy_net_state_dict': policy_net.state_dict(),
        #         'target_net_state_dict': target_net.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'episode_rewards': episode_rewards,
        #         'episode_durations': episode_durations
        #     }, model_save_path)
            
        #     print(f"Saved checkpoint at episode {i_episode+1}")
        
        # if (i_episode)==0:
        #     # Save model parameters using PyTorch
        #     model_save_path = f"saved_models/sanity_check_model" + date.today().strftime('%Y-%m-%d') + ".pt"
        #     torch.save({
        #         'episode': i_episode,
        #         'policy_net_state_dict': policy_net.state_dict(),
        #         'target_net_state_dict': target_net.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'episode_rewards': episode_rewards,
        #         'episode_durations': episode_durations
        #     }, model_save_path)
            
        #     print(f"Saved checkpoint at episode {i_episode+1}")


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

# Plot final metrics
plt.figure(figsize=(12, 4))

# Save final metrics
final_metrics_path = f"final_metrics.pkl"
with open(final_metrics_path, 'wb') as f:
    pickle.dump({
        'episode_durations': episode_durations,
        'episode_rewards': episode_rewards
    }, f)