import random
from qlearning import QLearningAgent, State, Action


class QLearningAgentEpsScheduling(QLearningAgent):
    def get_action(self, state: State) -> Action:
        """
        Compute the action to take in the current state, including exploration.

        Exploration is done with epsilon-greey. Namely, with probability self.epsilon, we should take a random action, and otherwise the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        # TODO add epsilon-greedy exploration with a scheduling to reduce epsilon
        is_greedy = False

        self.epsilon = self.epsilon * 0.99
        
        if random.random() < self.epsilon:
            is_greedy = True

        return (
            random.choice(self.legal_actions)
            if is_greedy
            else self.get_best_action(state)
        )
