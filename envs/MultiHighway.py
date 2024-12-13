import gymnasium
from gymnasium import Env
import highway_env

from ray.rllib.env import MultiAgentEnv

from ray.tune.registry import register_env

from agentic.Agent import Agent


class MultiHighway(Env):
    def __init__(self,agents:list[Agent]):
        env=gymnasium.make(
            "highway-v0",
            render_mode="rgb_array",
            config={
                "controlled_vehicles": 5,  # Five controlled vehicles
                "vehicles_count": 0,      
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

        self.env=env
        self.agents=agents

    def step(self, action:dict):
        """Essentially action and observation masking for Highway and MultiAgentEnv envs"""
        actions_tuple = tuple()
        for key,value in action:
            actions_tuple=actions_tuple+(value,)
        observation, reward, terminated, truncated, info=self.env.step(actions_tuple)
        print(observation, " before wrapping\n\n")
        new_obs={}
        for agent,o in zip(self.agents,observation):
            new_obs[agent.name]=o
        new_obs=gymnasium.spaces.Dict(*new_obs)
        print(new_obs," after wrapping\n\n")
        return new_obs, reward, terminated, truncated, info

    
    def reset(self):
        return self.env.reset(seed=0)
    
    def render(self):
        return self.env.render()
    
    def with_agent_groups(self):
        return super(MultiAgentEnv).with_agent_groups()