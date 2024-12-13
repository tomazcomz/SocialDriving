from typing import Dict, List
import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from torchsummary import summary



FCNET_HIDDENS = [32, 32]
LSTM_CELL_SIZE = 128

INPUT_SHAPE=[1,60,30]

CONV_CONFIG=[2,[12,6],1]

CONV_OUT=[49, 25, 6]


class BaselineTorchModel(TorchModelV2,nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        name: str,):
        super(BaselineTorchModel, self).__init__(obs_space,action_space,num_outputs,{},name)
        nn.Module.__init__(self)
        filters = CONV_CONFIG

        camadas=[]
        in_channels = 1
        out_channels, kernel, stride = filters

        self._convs=nn.Conv2d(in_channels,
                    out_channels,
                    kernel,
                    stride,)
        camadas.append(self._convs)
        camadas.append(nn.Flatten())
        
        in_size = CONV_OUT[0]*CONV_OUT[1]*CONV_OUT[2]
        out_size=FCNET_HIDDENS[0]
        self._fc_o=nn.Linear(
            in_size,
            out_size,
        )
        in_size = out_size
        camadas.append(self._fc_o)

        out_size=FCNET_HIDDENS[1]
        self._fc_d=nn.Linear(
            in_size,
            out_size,
        )
        camadas.append(self._fc_d)
        in_size = out_size

        self._extractor = nn.Sequential(*camadas)

        # If our num_outputs still unknown, we need to do a test pass to
        # figure out the output dimensions. This could be the case, if we have
        # the Flatten layer at the end.
        if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
            dummy_in = (
                torch.from_numpy(self.obs_space.sample())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            dummy_out = self._extractor(dummy_in)
            self.num_outputs = dummy_out.shape[1]

        vf_layers=[]

        self._val_head=nn.Linear(in_size,1)
        vf_layers.append(self._val_head)
        vf_layers.append(nn.Flatten())
        self._value_branch=nn.Sequential(*vf_layers)

        pol_layers=[]
        self._pol_head=nn.Linear(in_size,self.num_outputs)
        pol_layers.append(self._pol_head)
        self._pol_branch=nn.Sequential(*pol_layers)


        """summary(self._extractor,[1,60,30])
        summary(self._pol_branch)
        summary(self._value_branch)"""
        #print(self._extractor)

        


    def forward(self, state):
        self._features = state
        self._features = self._features.permute(0, 2, 1, 3)
        conv_out = F.softplus(self._convs(self._features))
        #conv_out=conv_out.permute(0,2,3,1)
        conv_out=nn.Flatten()(conv_out)
        self._features=F.softplus(self._fc_d(F.softplus(self._fc_o(conv_out))))
        model_out=self.policy()
        return model_out

    def policy(self):
        assert self._features is not None, "must call forward() first"
        return F.softmax(self._pol_branch(self._features))

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        val=F.tanh(self._value_branch(self._features))
        while len(val.shape)>1:
            val=val[0]
        return val