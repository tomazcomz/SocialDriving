from agentic.Agent import Agent
import torch
from torch import nn
import gymnasium as gym

class BaselineAgent(Agent):
    def __init__(self,
        name:str, 
        model:nn.Module,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        device):
        super(BaselineAgent,self).__init__(name,model,obs_space,action_space,num_outputs)
        self.device=device

    def action(self, state):
        # Did not implement e-greedy porque assumimos que há exploração
        self.model.to(self.device)
        with torch.no_grad():
            state = state.flatten()
            state = torch.tensor(state, device=self.device)
            return torch.argmax(self.model.forward(state)).item()
