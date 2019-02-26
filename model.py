#Based on the Exercise of Deep Q-Learning to solve OpenAI Gyms LunarLander environment.

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, isDuelingDQN=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            isDuelingDQN (bool): if 'True', then use Duelling Q-network as basis for Duelling DQN
        """
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.isDuelingDQN = isDuelingDQN
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.state_value = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # See also discussion on how to merge the two streams for a Dueling DQN:
        # https://knowledge.udacity.com/questions/25183
        if self.isDuelingDQN:
            #advantage value + state value - average advantage
            adv = self.fc3(x)
            avg_adv = adv.mean().expand_as(adv)
            #avg_adv = adv.mean(dim=1, keepdim=True)
            return self.fc3(x) + self.state_value(x) - avg_adv
        
        else:
            return self.fc3(x)
