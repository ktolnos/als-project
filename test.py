from typing import Dict, Optional
import numpy as np
import math

from preloaded import *


# === AVAILABLE PRELOADED ENVIRONMENT FUNCTIONS ===
#
# GridworldState((row, col)) -> GridworldState
#     constructs a state from a tuple of row and column
#
# get_start_state() -> GridworldState
#     returns the starting state in the lower left corner at (2, 0)
# 
# class GridworldAction(enum.Enum):
#     UP = (-1, 0)
#     DOWN = (1, 0)
#     LEFT = (0, -1)
#     RIGHT = (0, 1)
#
# get_all_actions() -> List[GridworldAction]
#     returns a list of all possible actions
#
# is_terminal_state(state: GridworldState) -> bool
#     returns True if the state is a terminal state
#
# value_function(state: GridworldState) -> float
#     returns the value of the state
#
# Gridworld() -> Gridworld
#     constructs a Gridworld environment
# gridworld.get_next_state(state: GridworldState, action: GridworldAction) -> GridworldState
#     returns the next state reached after taking the given action in the given state
#


class SearchNode(object):
    """
    This represents a node in the MCTS tree.
    """

    def __init__(self, state: GridworldState, parent: Optional["SearchNode"] = None):
        """
        Initialize a search node.
        :param state: The game state represented by this node.
        :param parent: The parent node of this node, or None if this is the root node.
        """

        self.state = state
        self.children: Dict[GridworldAction, SearchNode] = {}
        self.value = 0
        self.visits = 0
        self.parent = parent

    def __str__(self):
        return f"Node({self.state}, {self.value}, {self.visits})"

    def select(self, env: Gridworld, uct_coefficient: float) -> "SearchNode":
        """
        Select the best descendent node to explore.
        """

        best_action, best_action_score = None, float("-inf")

        for action in get_all_actions():
            score = 0
            if action not in self.children:
                score = uct_coefficient * math.sqrt(math.log(self.visits + 1) / 1)
            else:
                ns = self.children[action]
                score = ns.value / ns.visits + uct_coefficient * math.sqrt(
                    (math.log(self.visits) + 1) / (ns.visits + 1))
            if score > best_action_score:
                best_action = action
                best_action_score = score

        if best_action not in self.children:
            self.expand(best_action, env)

        return self.children[best_action]

    def expand(self, action: GridworldAction, env: Gridworld) -> "SearchNode":
        """
        Expand the tree by adding a new child node to this node.
        :param action: The action to take to reach the new child node.
        """

        assert action not in self.children
        if is_terminal_state(self.state):
            return self

        new_state = env.get_next_state(self.state, action)
        child = SearchNode(new_state, parent=self)
        self.children[action] = child
        return child

    def backpropagate(self, value: float) -> None:
        """
        Backpropagate the value of a leaf node to the root node.
        """

        self.visits += 1
        self.value += value

        if self.parent is not None:
            self.parent.backpropagate(value)


def run_search(
        root_node: SearchNode,
        num_simulations: int,
        uct_coefficient: float = 1.0,
) -> Dict[GridworldAction, int]:
    """
    Run MCTS search given a root node for a certain number of simulations.
    :param root_node: The root node of the MCTS tree.
    :param num_simulations: The number of simulations to run; it will always be at
        least 1.
    :param uct_coefficient: The PUCT coefficient to use for the search.
    :return: A dictionary mapping actions to the number of times they were selected
        during the search.
    """

    env = Gridworld()
    for _ in range(num_simulations):
        node = root_node.select(env, uct_coefficient)
        while not is_terminal_state(node.state) and node.visits > 0:
            node = node.select(env, uct_coefficient)
        value = value_function(node.state)
        node.backpropagate(value)

    return {action: child.visits for action, child in root_node.children.items()}


def select_action(
        visit_counts: Dict[GridworldAction, int],
) -> GridworldAction:
    """
    Select the best action to take given the visit counts of each action.
    :param visit_counts: A dictionary mapping actions to the number of times they were
        selected during the search.
    :return: The most visited action.
    """

    # DON'T MODIFY THIS FUNCTION

    best_action, best_visits = None, float("-inf")
    for action, visits in visit_counts.items():
        if visits > best_visits:
            best_action = action
            best_visits = visits
    return best_action