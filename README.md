# Social Driving

This is the official repository for the Tópicos em Sistemas Ineligentes final delivery.
As it stands, the work presents a from scratch implementation of a Multi-Agent Proximal Policy Optimization (MAPPO) and a Multi-Agent Deep Q-Network(MADQN). These are applied to the Highway and Intersection Environments respectively.

The MADQN scenario also implements the Basical Social Influence reward scheme.

---

## Installation Guide

Follow the steps below to set up and run the Social Driving project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/tomazcomz/SocialDriving/tree/main
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

## Running Considerations

To execute the Highway Env with the MAPPO run the script as follows:

`python -m MA_highway  --num_agents=[n] --num_episodes=[m]`

All available parameters are found in util/default_args.py, including model loading settings.

----------------------------------------------------------------------------------------------------------------------------------------------




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
  url = {https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html%7D},
  note = {Accessed: 2024-12-15}
}
```

## Contributors
David Mesquita Scarin 202108314

Tomás Amorim Azevedo 202107313

Pedro Rafael Castro Sousa 202108383

Isabel Antónia Costa Brito 202107271

