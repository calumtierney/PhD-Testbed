"""Gate 6: Hughes et al. 2011's 1000-point variance split (about 80% angle
noise, about 20% geometric parameters, 0.1-3.8% each), through the design path."""
import numpy as np
import pytest

from instruments.laser_tracker import LaserTracker
from instruments.npl_model import INDEX, NPLErrorParameters
from network.variance_split import parameter_variance_split, random_points

RESIDUAL = NPLErrorParameters.hughes_2011_residual()


def test_parameters_carry_about_a_fifth_of_the_variance_and_the_angle_sensors_four_fifths():
    """The paper's two headline numbers, from the Table 2-3 uncertainties with no
    correlations and the Table 4 noise: 19.2% and 80.0%, against 'about 20%' and
    'approximately 80%'. Stable across seeds and across how the random angles
    are drawn."""
    for distribution in ("uniform_angle", "uniform_direction"):
        for seed in (0, 1, 2):
            split = parameter_variance_split(RESIDUAL, seed=seed, distribution=distribution)
            assert 0.17 < split["parameters"] < 0.22
            assert 0.78 < split["angle_noise"] < 0.83
            assert split["range_noise"] < 0.01
            assert split["parameters"] + split["noise"] == pytest.approx(1.0, abs=1e-12)


def test_per_parameter_shares_add_up_and_are_individually_small_except_the_elevation_offset():
    split = parameter_variance_split(RESIDUAL, seed=0)
    assert abs(split["additivity_error"]) < 1e-10  # diagonal covariance: the shares are additive
    assert sum(split["per_parameter"].values()) == pytest.approx(split["parameters"], rel=1e-10)
    shares = split["per_parameter"]
    assert max(shares, key=shares.get) == "a_E0_rad"      # the elevation zero offset dominates
    assert shares["a_E0_rad"] > 0.05 and all(v < 0.025 for k, v in shares.items() if k != "a_E0_rad")
    assert all(v > 0.0 for v in shares.values())


def test_no_parameter_uncertainty_means_no_parameter_share():
    zero_covariance = NPLErrorParameters(RESIDUAL.values, np.zeros((16, 16)))
    assert parameter_variance_split(zero_covariance)["parameters"] == 0.0


def test_random_points_respect_the_stated_ranges():
    points = random_points(500, np.random.default_rng(0))
    d = np.linalg.norm(points, axis=1)
    elevation = np.rad2deg(np.arcsin(points[:, 2] / d))
    assert d.min() >= 1.0 and d.max() <= 5.0 and elevation.min() >= -60.0 and elevation.max() <= 77.0
    with pytest.raises(ValueError):
        random_points(5, np.random.default_rng(0), distribution="other")
