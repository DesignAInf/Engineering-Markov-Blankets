"""Agents that learn the topology of their own blanket.

``simulate_learner``  joint Bayesian filter over (eta, kappa) within an episode,
                      with passive, random-probe or one-step-EFE announcements.
``run_episode``       one episode for many worlds in parallel, with a prior over
                      kappa carried across episodes and a per-world choice of
                      boundary configuration (visible = loop active,
                      hidden = loop cut). Used by the self-engineering experiment.
"""
from __future__ import annotations

import numpy as np

from .core import SEED, decide

DEFAULT_GRID = np.round(np.arange(0.0, 0.91, 0.1), 2)


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) along the last axis."""
    return -np.sum(p * np.log(p + 1e-300), axis=-1)


def entropy_joint(w: np.ndarray) -> np.ndarray:
    """Entropy of a (N, 2, K) joint belief over (eta, kappa)."""
    return -np.sum(w * np.log(w + 1e-300), axis=(1, 2))


def simulate_learner(N: int, T: int, rho: float, kappa_true: float, grid=DEFAULT_GRID,
                     mode: str = "passive", eps: float = 0.0, beta: float = 1.0,
                     seed: int = SEED) -> dict:
    """Within-episode joint (eta, kappa) learner in the closed loop.

    mode : "passive" (announce argmax), "random" (random announcement with
           probability eps), or "efe" (argmin over announcements of
           G(x) = [1 - q(eta = x)] - beta * expected information gain on (eta, kappa)).
    """
    rng = np.random.default_rng(seed)
    grid = np.asarray(grid)
    K = len(grid)
    kap = grid[None, None, :]
    etas = np.array([0, 1])[None, :, None]
    eta = rng.integers(0, 2, N)
    w = np.full((N, 2, K), 1.0 / (2 * K))
    a_prev = None
    ann_correct, n_probe = [], []
    for t in range(T):
        s1 = np.where(rng.random(N) < rho, eta, 1 - eta)
        ind2 = np.where(rng.random(N) < rho, eta, 1 - eta)
        u_copy = rng.random(N)
        s2 = ind2 if t == 0 else np.where(u_copy < kappa_true, a_prev, ind2)

        lik = np.where(s1[:, None, None] == etas, rho, 1 - rho)
        ind_lik = np.where(s2[:, None, None] == etas, rho, 1 - rho)
        if t == 0:
            lik = lik * ind_lik
        else:
            same = (s2 == a_prev)[:, None, None]
            lik = lik * (kap * same + (1 - kap) * ind_lik)
        w = w * lik
        w /= w.sum(axis=(1, 2), keepdims=True)

        q1 = w[:, 1, :].sum(axis=1)
        greedy = decide(np.log(q1 + 1e-300) - np.log(1 - q1 + 1e-300), rng)
        u_eps = rng.random(N)
        r_ann = rng.integers(0, 2, N)
        if mode == "passive":
            a = greedy
        elif mode == "random":
            a = np.where(u_eps < eps, r_ann, greedy)
        elif mode == "efe":
            H0 = entropy_joint(w)
            G = np.zeros((N, 2))
            for x in (0, 1):
                risk = 1 - (q1 if x == 1 else 1 - q1)
                EH = np.zeros(N)
                for o1 in (0, 1):
                    l1 = np.where(o1 == etas, rho, 1 - rho)
                    for o2 in (0, 1):
                        l2 = kap * (o2 == x) + (1 - kap) * np.where(o2 == etas, rho, 1 - rho)
                        joint = w * l1 * l2
                        po = joint.sum(axis=(1, 2))
                        post = joint / po[:, None, None]
                        EH += po * entropy_joint(post)
                G[:, x] = risk - beta * (H0 - EH)
            a = np.argmin(G, axis=1)
            tie = G[:, 0] == G[:, 1]
            a[tie] = greedy[tie]
        else:
            raise ValueError("mode must be 'passive', 'random' or 'efe'")
        n_probe.append(np.mean(a != greedy))
        ann_correct.append(np.mean(a == eta))
        a_prev = a

    q1 = w[:, 1, :].sum(axis=1)
    final_dec = (q1 > 0.5).astype(int)
    pk = w.sum(axis=1)
    kmean = pk @ grid
    p_true = np.where(eta == 1, q1, 1 - q1)
    return dict(
        final_accuracy=float(np.mean(final_dec == eta)),
        announcement_accuracy=float(np.mean(ann_correct)),
        probe_rate=float(np.mean(n_probe)),
        kappa_abs_error=float(np.mean(np.abs(kmean - kappa_true))),
        kappa_entropy=float(np.mean(entropy(pk))),
        log_loss=float(np.mean(-np.log(np.clip(p_true, 1e-300, 1)))),
        lockin=float(np.mean(p_true < 0.01)),
    )


def run_episode(rng: np.random.Generator, prior_k: np.ndarray, kappa_true: np.ndarray,
                visible: np.ndarray, T: int = 30, rho: float = 0.60, grid=DEFAULT_GRID):
    """One episode for N worlds in parallel.

    prior_k    : (N, K) belief over kappa at the start of the episode.
    kappa_true : (N,) true loop strength of each world.
    visible    : (N,) bool; False means the agent withholds its announcements
                 (the loop is cut) for this episode.

    Returns (correct (N,), posterior over kappa (N, K), episode log-likelihood
    over kappa (N, K), normalised to max 0).
    """
    grid = np.asarray(grid)
    K = len(grid)
    N = len(kappa_true)
    kap = grid[None, None, :]
    etas = np.array([0, 1])[None, :, None]
    eta = rng.integers(0, 2, N)
    w = 0.5 * prior_k[:, None, :] * np.ones((N, 2, K))
    w /= w.sum(axis=(1, 2), keepdims=True)
    loglik = np.zeros((N, 2, K))
    a_prev = None
    for t in range(T):
        s1 = np.where(rng.random(N) < rho, eta, 1 - eta)
        ind2 = np.where(rng.random(N) < rho, eta, 1 - eta)
        u = rng.random(N)
        s2 = ind2 if t == 0 else np.where(visible & (u < kappa_true), a_prev, ind2)
        l1 = np.where(s1[:, None, None] == etas, rho, 1 - rho)
        ind_lik = np.where(s2[:, None, None] == etas, rho, 1 - rho)
        if t == 0:
            l2 = ind_lik * np.ones((1, 1, K))
        else:
            same = (s2 == a_prev)[:, None, None]
            loop = kap * same + (1 - kap) * ind_lik
            l2 = np.where(visible[:, None, None], loop, ind_lik * np.ones((1, 1, K)))
        lik = l1 * l2
        loglik += np.log(lik)
        w = w * lik
        w /= w.sum(axis=(1, 2), keepdims=True)
        q1 = w[:, 1, :].sum(axis=1)
        a_prev = decide(np.log(q1 + 1e-300) - np.log(1 - q1 + 1e-300), rng)
    correct = (a_prev == eta)
    post_k = w.sum(axis=1)
    m = loglik.max(axis=(1, 2), keepdims=True)
    lk = np.log(np.exp(loglik - m).sum(axis=1)) + m[:, 0, :]
    lk -= lk.max(axis=1, keepdims=True)
    return correct, post_k, lk
