"""Closed-form quantities used in the propositions of the paper."""
from __future__ import annotations

import numpy as np

from .core import logit


def reff_from_alpha(alpha_prev, rho: float, kappa: float):
    """Step-matched open-loop reliability (Eq. reff):
    rho_eff_t = kappa * alpha_{t-1} + (1 - kappa) * rho."""
    return kappa * np.asarray(alpha_prev) + (1 - kappa) * rho


def cov_echo(alpha_prev, kappa: float):
    """Proposition 2: Cov_A(s2_t, a_{t-1} | eta) = kappa * Var(a_{t-1} | eta),
    with Var = alpha (1 - alpha) for a binary announcement of accuracy alpha."""
    alpha_prev = np.asarray(alpha_prev)
    return kappa * alpha_prev * (1 - alpha_prev)


def garbling_flip_prob(reff, rho: float):
    """Proposition 3: flip probability turning a reliability-reff sensor into a
    reliability-rho sensor (requires reff >= rho > 1/2)."""
    reff = np.asarray(reff, dtype=float)
    if np.any(reff < rho):
        raise ValueError("garbling requires reff >= rho")
    return (reff - rho) / (2 * reff - 1)


def naive_drift_when_wrong(rho: float, kappa, r):
    """Proposition 4: expected one-step change (nats) of the naive agent's
    log-odds in favour of its current, wrong announcement."""
    return kappa * logit(r) - (2 * rho - 1) * (logit(rho) + (1 - kappa) * logit(r))


def kappa_star(rho: float, r):
    """Proposition 4: loop strength above which wrong beliefs self-reinforce."""
    return (2 * rho - 1) * (logit(rho) + logit(r)) / (2 * rho * logit(r))


def kl_bernoulli(p: float, q: float) -> float:
    """KL(Bern(p) || Bern(q)) in nats."""
    return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))
