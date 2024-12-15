# Social Driving

This is the official repository for the Tópicos em Sistemas Ineligentes final delivery.
As it stands, the work presents a from scratch implementation of a Multi-Agent Proximal Policy Optimization (MAPPO) and a Multi-Agent Deep Q-Network(MADQN). These are applied to the Highway and Intersection Environments respectively.

The MADQN scenario also implements the Basical Social Influence reward scheme.

---

## Installation Guide

Follow the steps below to set up and run the Social Driving project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/tomazcomz/SocialDriving.git
   ```

2. Navigate to the project directory:
   ```bash
   cd SocialDriving
   ```

3. Create the required Conda environment using the provided `sih.yml` file (!watchout for cuda versions don't mess up your drivers!):
   ```bash
   conda env create -f sih.yml
   ```

4. Activate the Conda environment:
   ```bash
   conda activate sih
   ```

---

## Running MAPPO

To execute the multi-agent setting in Highway Env with the MAPPO run the script as follows:

`python -m MA_highway  --num_agents=[n] --num_episodes=[m]`

All available parameters are found in util/default_args.py, including model loading settings.

----------------------------------------------------------------------------------------------------------------------------------------------

## Running MADQN

To execute the multi-agent setting in Highway Env with the DQN model run the script as follows with or without the arguments described below:

```python decentralized_dqn.py```

argument list:

```
--load_model: load an existing model, by default False
--only_agents: choose to train only ego vehicles and no other cars, by default False
--render: whether or not to visually display the environment, by default False
--lr: set the desired learning rate, by default 1e-5
--batch_size: set the batch size for learning, by default 256
--max_eps: set the maximum number of episodes, by default 100000
```

## References:

Highway Env:
```tex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
```

The DQN implementation follows thoroughly: 

```tex
@misc{pytorch_rl_tutorial,
  author = {Adam Paszke},
  title = {Reinforcement Learning (Q-Learning) Tutorial},
  year = {n.d.},
  url = {https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html},
  note = {Accessed: 2024-12-15}
}
```

## Contributors
David Mesquita Scarin 202108314

Tomás Amorim Azevedo 202107313

Pedro Rafael Castro Sousa 202108383

Isabel Antónia Costa Brito 202107271

