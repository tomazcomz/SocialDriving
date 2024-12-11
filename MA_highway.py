import gymnasium
import highway_env
import torch

from models.BaselineTorchModel import BaselineTorchModel

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print("using ", device)


def get_action(state,model:BaselineTorchModel):
    with torch.no_grad():
        return model.forward(state).max(1).indices.view(1,1)

def routine(num_episodes):
    env=gymnasium.make(
        "highway-v0",
        render_mode="rgb_array",
        config={
            "controlled_vehicles": 5,  # Five controlled vehicles
            "vehicles_count": 2,      
        }
    )

    env.unwrapped.config.update({
        "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (60, 30),
        "stack_size": 1,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
        },

        "action":{
            "type":"MultiAgentAction",
            "action_config": {
                "type": "ContinuousAction",
            }
        }
    })



    for ep in range(num_episodes):
        env.reset(seed=0)
        
