import torch

def compute_influence_reward(agents:list,state, next_state, actions:list):
    """Compute the influence reward for each agent."""
    influence_rewards = {}
    for i in range(len(agents)):
        other_agents = [j for j in range(len(agents)) if j != i]
        original_action_dists = {
            j: torch.softmax(agents[j](state[j]), dim=-1) for j in other_agents
        }
        counterfactual_dists = {j: [] for j in other_agents}

        for counterfactual_action in range(len(actions)):
            for j in other_agents:
                counterfactual_logits = agents[j](state[j]).clone()
                counterfactual_logits[0, counterfactual_action] += 1e-6
                counterfactual_dists[j].append(torch.softmax(counterfactual_logits, dim=-1))

        total_influence = 0.0
        for j in other_agents:
            counterfactual_mean_dist = torch.stack(counterfactual_dists[j]).mean(0)
            kl_div = torch.sum(
                original_action_dists[j] * torch.log(original_action_dists[j] / (counterfactual_mean_dist + 1e-9))
            )
            total_influence += kl_div.item()

        influence_rewards[i] = total_influence
    return influence_rewards