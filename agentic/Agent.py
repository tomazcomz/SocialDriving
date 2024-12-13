from torch import nn

class Agent:
    def __init__(self,name:str,model:nn.Module):
        self.model=model
        self.name=name

    def get_model(self):
        return self.model
    
    def action(self,state):
        pass