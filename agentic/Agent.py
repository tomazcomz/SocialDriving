from torch import nn
import gymnasium as gym

class Agent:
    def __init__(self,name:str,
        model:nn.Module,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        ):
        self.model=model(obs_space,action_space,num_outputs,{},name)
        self.name=name

    

    def get_model(self):
        return self.model
    
    def action(self,state):
        pass