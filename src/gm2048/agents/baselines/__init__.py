"""Agents de référence, sans apprentissage."""

from gm2048.agents.baselines.greedy import GreedyAgent
from gm2048.agents.baselines.heuristic import HeuristicAgent, HeuristicWeights
from gm2048.agents.baselines.random_agent import RandomAgent

__all__ = ["GreedyAgent", "HeuristicAgent", "HeuristicWeights", "RandomAgent"]
