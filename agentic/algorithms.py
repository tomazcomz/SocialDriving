import torch
from torch import multiprocessing,nn

import gymnasium as gym

from torch import optim

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

    def __repr__(self):
        return f"ReplayMemory(capacity={len(self.memory)}, contents={list(self.memory)})"



class MAPPO():

    def __init__(self,input_env:gym.Env,device:torch.device,config):

        self.device=device
        self.config=config

        # When calling MAPPO, we should pass input_env=MultiHighway.env
        self.env=input_env

    
    def rollout(self,env:gym.Env,max_steps:int,agents:list[Agent],device:torch.device)->ReplayMemory:
        states, info = env.reset()
        #print(states)
        agent_memories = {i : ReplayMemory(10000) for i in range(len(states))}
        terminated=False
        actions_tuple = tuple()
        episode_reward=0
        steps=0
        while not terminated and steps<max_steps:
            steps+=1
            for i_state in range(len(states)):
                with torch.no_grad():
                    actions_tuple = actions_tuple + (agents[i_state].action([states[i_state]]), )

            observation, reward, terminated, truncated, info = env.step(actions_tuple)
            episode_reward+=reward
            done = terminated or truncated
            next_states = observation
            for i_agent in range(len(states)):
                state = states[i_agent]
                if not done:
                    next_state = torch.tensor([next_states[i_agent]], device=device)
                else:
                    next_state = None
                agent_memories[i_agent].push(torch.tensor([state], device=device), torch.tensor([actions_tuple[i_agent]], device=device), next_state, torch.tensor([reward], device=device))        
            states = next_states
            #if self.config.render:
                #env.render()
        return agent_memories,steps


    

    def train(self,num_episodes:int,max_steps:int,agents:list[Agent]):
        episode_reward_mean_list = []
        actor_losses={}
        critic_losses={}
        episode_durations = []
        episode_rewards = []
        for ep in range(num_episodes):
            print(f"Starting episode {ep}")
            memories,t=self.rollout(self.env,max_steps,agents,self.device)
            optimizers = {agent: optim.Adam(agent.model.parameters(), lr=self.config.lr) for agent in agents}
            old_policies={agent:agent.model for agent in agents}
            for replay,agent in zip(memories.values(),agents):
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*[(t.state, t.action, t.next_state, t.reward) for t in replay.memory]))

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
                action_batch_indices = action_batch.argmax(dim=1).unsqueeze(1)
                state_action_values = agent.model.forward(state_batch).gather(1, action_batch_indices)
                

                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1).values
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(len(replay.memory), device=self.device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = agent.model.value_function(non_final_next_states)
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch

                



                advantage=state_action_values.squeeze(1)-expected_state_action_values
                advantage=advantage.unsqueeze(1)

                critic_criterion = nn.KLDivLoss()
                critic_loss = critic_criterion(state_action_values, expected_state_action_values.unsqueeze(0))

                #Calculating PPO Loss

                old_probs=old_policies[agent].forward(state_batch)
                new_probs = agent.model.forward(state_batch)
                ratios = new_probs / old_probs

                # Unclipped part of the surrogate loss function
                surr1 = advantage*ratios

                # Clipped part of the surrogate loss function
                surr2 = torch.clamp(ratios, 1 - self.config.clip, 1 + self.config.clip) * advantage

                # Update actor network: loss = min(surr1, surr2)
                actor_loss = -torch.min(surr1, surr2).mean()

                # Optimization: backward, grad clipping and optimization step
                actor_loss.backward(retain_graph=True)
                critic_loss.backward(retain_graph=True)
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(agent.model._pol_branch.parameters(), self.config.grad_clip)
                torch.nn.utils.clip_grad_norm_(agent.model._value_branch.parameters(), self.config.grad_clip)
                optimizers[agent].step()
                optimizers[agent].zero_grad()

                old_policies[agent]=agent.model

                actor_losses[agent]=actor_loss
                critic_losses[agent]=critic_loss

                print(f"Done optimization with agent {agent.name}")
            print(f"Finished episode {ep}")

            
            """
            episode_durations.append(t + 1)
            episode_rewards.append(episode_reward)

            writer.add_scalar('Training/Episode Duration', t + 1, ep)
            writer.add_scalar('Training/Episode Reward', episode_reward, ep)
            for i_agent in range(len(agents)):
                if losses[i_agent] is not None:
                    writer.add_scalar(f'Training/Loss Agent {i_agent}', losses[i_agent].item(), ep)
            
                # Save models and data every 1000 episodes
                if (ep + 1) % 500 == 0:
                    # Save model parameters using PyTorch
                    for n_agent in range(len(agents)):
                        model_save_path = "saved_models/"+"model_"+model_name+"agent "+str(n_agent)+" "+date.today().strftime('%Y-%m-%d')+"_episode_"+str(ep)+".pt"
                        torch.save({
                            'episode': ep,
                            'agent_policy_net_state_dict': agents[n_agent].state_dict(),
                            'optimizer_state_dict': optimizers[n_agent].state_dict(),
                            'episode_rewards': episode_rewards,
                            'episode_durations': episode_durations
                        }, model_save_path)
                           
                        print(f"Saved checkpoint at episode {ep+1}")
                    
                if (ep)==0:
                    # Save model parameters using PyTorch
                    model_save_path = f"saved_models/sanity_check_model" + date.today().strftime('%Y-%m-%d') + ".pt"
                    torch.save({
                    'episode': ep,
                        'policy_net_state_dict': agents[0].state_dict(), #first agent picked arbitrarily
                        'optimizer_state_dict': optimizers[0].state_dict(),
                        'episode_rewards': episode_rewards,
                        'episode_durations': episode_durations
                    }, model_save_path)
                       
                    print(f"Saved checkpoint at episode {ep+1}")
"""