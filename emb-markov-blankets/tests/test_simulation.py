import numpy as np

from emb.core import fixed_point_reff, simulate
from emb.learner import DEFAULT_GRID, run_episode, simulate_learner

N, T, RHO = 20_000, 12, 0.6


def test_kappa_zero_conditions_coincide():
    reff = np.full(T, RHO)
    a = simulate("A", "naive", N, T, RHO, 0.0, reff)["final"]["accuracy"]
    b = simulate("A", "aligned", N, T, RHO, 0.0, reff)["final"]["accuracy"]
    assert a == b  # same random stream, identical likelihoods


def test_stepwise_marginals_match_between_A_and_B():
    """Proposition 1: per-step P(s2_t = eta) coincides in A and B."""
    kappa = 0.5
    reff = fixed_point_reff(N, T, RHO, kappa)
    sA = simulate("A", "naive", N, T, RHO, kappa, reff)["s2acc"]
    sB = simulate("B", "naive", N, T, RHO, kappa, reff, seed=1)["s2acc"]
    assert np.max(np.abs(sA - sB)) < 0.02


def test_naive_overconfident_in_loop_but_calibrated_in_open_loop():
    kappa = 0.5
    reff = fixed_point_reff(N, T, RHO, kappa)
    fa = simulate("A", "naive", N, T, RHO, kappa, reff)["final"]
    fb = simulate("B", "naive", N, T, RHO, kappa, reff, seed=1)["final"]
    fal = simulate("A", "aligned", N, T, RHO, kappa, reff)["final"]
    assert fa["mean_confidence"] - fa["accuracy"] > 0.05
    assert abs(fb["mean_confidence"] - fb["accuracy"]) < 0.02
    assert abs(fal["mean_confidence"] - fal["accuracy"]) < 0.02
    assert fb["accuracy"] > fal["accuracy"] > fa["accuracy"]  # Prop. 3 and Result 2


def test_hidden_boundary_leaves_kappa_belief_unchanged():
    """Proposition 5: with announcements withheld, q(kappa) does not move."""
    rng = np.random.default_rng(0)
    K = len(DEFAULT_GRID)
    prior = rng.dirichlet(np.ones(K), size=200)
    kappa = rng.choice(DEFAULT_GRID, size=200)
    _, post, _ = run_episode(rng, prior, kappa, np.zeros(200, bool), T=T, rho=RHO)
    assert np.allclose(post, prior, atol=1e-10)


def test_learner_runs_in_all_modes():
    for mode in ("passive", "random", "efe"):
        r = simulate_learner(500, 8, RHO, 0.5, mode=mode, eps=0.2, beta=1.0)
        assert 0.5 <= r["final_accuracy"] <= 1.0
