import torch

def compute_influence_reward(agents: list, state, actions: list, device):
    """Compute the influence reward for each agent."""
    influence_rewards = {}
    n_actions = 3  # Action space size (discrete actions)

    for i in range(len(agents)):
        other_agents = [j for j in range(len(agents)) if j != i]

        # Compute original action distributions for other agents
        original_action_dists = {
            j: torch.softmax(agents[j](torch.tensor(state[j].flatten(), dtype=torch.float, device=device).unsqueeze(0)), dim=-1)
            for j in other_agents
        }

        # Prepare counterfactual distributions
        counterfactual_dists = {j: [] for j in other_agents}

        # Generate counterfactual logits
        for counterfactual_action in range(n_actions):
            for j in other_agents:
                # Compute logits for the current state
                logits = agents[j](torch.tensor(state[j].flatten(), dtype=torch.float, device=device).unsqueeze(0)).squeeze(0).clone()
                
                # Apply counterfactual adjustment
                logits[counterfactual_action] += 1e-6  # Small perturbation to simulate counterfactual
                
                # Compute action distribution with counterfactual logits
                counterfactual_dists[j].append(torch.softmax(logits, dim=-1))

        # Calculate influence rewards using KL divergence
        total_influence = 0.0
        for j in other_agents:
            # Average the counterfactual distributions
            counterfactual_mean_dist = torch.stack(counterfactual_dists[j]).mean(0)

            # Compute KL divergence
            kl_div = torch.sum(
                original_action_dists[j] * torch.log(original_action_dists[j] / (counterfactual_mean_dist + 1e-9))
            )
            total_influence += kl_div.item()

        # Assign the influence reward for agent i
        influence_rewards[i] = total_influence

    return influence_rewards