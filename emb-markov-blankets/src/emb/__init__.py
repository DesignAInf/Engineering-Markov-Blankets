"""Engineering Markov Blankets: when sensory states descend from active states.

Reference implementation accompanying the paper by Luca M. Possati.
"""
from .core import SEED, decide, fixed_point_reff, logit, simulate
from .learner import DEFAULT_GRID, run_episode, simulate_learner
from .theory import (cov_echo, garbling_flip_prob, kappa_star, kl_bernoulli,
                     naive_drift_when_wrong, reff_from_alpha)

__version__ = "0.9.0"
__author__ = "Luca M. Possati"
__email__ = "l.m.possati@utwente.nl"

__all__ = [
    "SEED", "decide", "fixed_point_reff", "logit", "simulate",
    "DEFAULT_GRID", "run_episode", "simulate_learner",
    "cov_echo", "garbling_flip_prob", "kappa_star", "kl_bernoulli",
    "naive_drift_when_wrong", "reff_from_alpha",
]
