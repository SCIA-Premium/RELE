import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Hyperparameters
GAMMA = 0.99
def compute_loss(
    model: nn.Module,
    state: torch.Tensor, 
    next_state: torch.Tensor,
    reward: torch.Tensor,
    action: torch.Tensor,
    done: torch.Tensor,
):
    """
    Compute the DQN agent loss
    """
    # Compute the Q values for the current state
    q_values = model(state)
    # Compute the Q values for the next state
    next_q_values = model(next_state)
    # Compute the Q value for the action taken
    q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    # Compute the max Q value at the next state (for the next action)
    next_q_value = next_q_values.max(1)[0]
    # Compute the expected Q value
    expected_q_value = reward + GAMMA * next_q_value * (1 - done)
    # Return the loss (MSE between the expected and the actual Q values)
    return F.mse_loss(q_value, expected_q_value)