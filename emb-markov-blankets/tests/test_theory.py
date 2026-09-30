import numpy as np
import pytest

from emb.theory import (cov_echo, garbling_flip_prob, kappa_star, kl_bernoulli,
                        naive_drift_when_wrong, reff_from_alpha)


def test_drift_is_zero_at_kappa_star():
    for rho in (0.55, 0.6, 0.7):
        for r in (0.62, 0.68, 0.75):
            ks = kappa_star(rho, r)
            assert abs(naive_drift_when_wrong(rho, ks, r)) < 1e-12


def test_drift_sign_around_threshold():
    rho, r = 0.6, 0.68
    ks = kappa_star(rho, r)
    assert naive_drift_when_wrong(rho, ks - 0.05, r) < 0
    assert naive_drift_when_wrong(rho, ks + 0.05, r) > 0


def test_garbling_flip_gives_target_reliability():
    rho = 0.6
    for reff in (0.6, 0.65, 0.8):
        phi = garbling_flip_prob(reff, rho)
        assert 0 <= phi < 0.5
        assert abs(reff * (1 - phi) + (1 - reff) * phi - rho) < 1e-12


def test_garbling_requires_reff_ge_rho():
    with pytest.raises(ValueError):
        garbling_flip_prob(0.55, 0.6)


def test_reff_and_cov():
    assert reff_from_alpha(0.8, 0.6, 0.5) == pytest.approx(0.7)
    assert cov_echo(0.5, 0.5) == pytest.approx(0.125)
    assert cov_echo(1.0, 0.5) == pytest.approx(0.0)


def test_kl_positive():
    assert kl_bernoulli(0.6, 0.4) > 0
    assert kl_bernoulli(0.6, 0.6) == pytest.approx(0.0)
