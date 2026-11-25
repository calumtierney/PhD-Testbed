"""Gate 5: the network test of Hughes et al. (2011), reconstructed
(`network.hughes_network`) and fitted in estimate mode."""
from dataclasses import replace

import numpy as np
import pytest

from instruments.npl_model import BACK, FRONT, INDEX, N_PARAMETERS, NPLErrorParameters
from network.hughes_network import (
    BENCH_TARGETS, BIRDBATH_LINES, FLOOR_TARGETS, NETWORK_TEST_PRIOR, TARGET_POSITIONS_M, WALL_TARGETS,
    fit_network_test, hughes_2011_network_scenario, network_test_covariance, simulate_network_test,
)

TRUE = NPLErrorParameters.hughes_2011_uncompensated()
NETWORK = hughes_2011_network_scenario()


def correlation(network, params=TRUE, prior=NETWORK_TEST_PRIOR):
    covariance = network_test_covariance(network, params, prior).parameter_estimates[0].covariance
    s = np.sqrt(np.diag(covariance))
    return covariance / np.outer(s, s), s


def test_layout_matches_the_papers_description():
    ranges, elevations = NETWORK.ranges_and_elevations()
    measured = NETWORK.visible
    assert NETWORK.targets_m.shape == (15, 3) and len(NETWORK.station_poses) == 5
    assert len(WALL_TARGETS) == 8 and len(BENCH_TARGETS) == 6 and len(FLOOR_TARGETS) == 1
    assert sorted(WALL_TARGETS + BENCH_TARGETS + FLOOR_TARGETS) == list(range(1, 16))
    assert measured.sum() == 63 and NETWORK.n_readings == 126             # 126 of a possible 15 x 5 x 2 = 150
    assert ranges[measured].min() == pytest.approx(0.27, abs=0.01) and ranges[measured].max() == pytest.approx(6.4, abs=0.01)
    assert elevations[measured].max() <= 57.0 and elevations[measured].max() > 55.0 and elevations[measured].min() > -60.0
    assert np.mean(np.cos(2 * np.deg2rad(elevations[measured]))) == pytest.approx(0.69, abs=0.03)   # the paper's mean cos 2phi
    positions = np.array([p.position_m for p in NETWORK.station_poses])
    np.testing.assert_array_equal(positions[0], positions[1])             # position 1 is position 0, rotated
    assert not np.allclose(NETWORK.station_poses[0].orientation, NETWORK.station_poses[1].orientation)
    assert measured.sum(axis=0).min() >= 3                                # every nest seen from at least three positions
    assert positions[2][2] > positions[0][2] > positions[3][2] - 1e-9     # position 2 at full height; 3 on the bench; 4 lowest
    assert positions[4][2] == positions[:, 2].min()
    assert all(NETWORK.faces[s] == (FRONT, BACK) for s in range(5))


def test_the_two_birdbath_lines_are_collinear_with_the_right_trackers_inside_and_outside():
    positions = np.array([p.position_m for p in NETWORK.station_poses])
    for (a, b), external, internal in BIRDBATH_LINES:
        pa, pb = TARGET_POSITIONS_M[a - 1], TARGET_POSITIONS_M[b - 1]
        direction = (pb - pa) / np.linalg.norm(pb - pa)
        for station in (external, internal):
            offset = positions[station] - pa
            assert np.linalg.norm(offset - (offset @ direction) * direction) < 1e-9    # on the line of the two targets
            assert NETWORK.visible[station, a - 1] and NETWORK.visible[station, b - 1]  # and sees both
        along = lambda station: (positions[station] - pa) @ direction
        assert 0.0 < along(internal) < np.linalg.norm(pb - pa)                         # between the targets
        assert along(external) < 0.0 or along(external) > np.linalg.norm(pb - pa)      # outside them
    (a, b), _, _ = BIRDBATH_LINES[0]
    horizontal = TARGET_POSITIONS_M[b - 1] - TARGET_POSITIONS_M[a - 1]
    assert abs(horizontal[2]) < 1e-12 and np.linalg.norm(horizontal) == pytest.approx(2.0, abs=0.05)
    (a, b), _, _ = BIRDBATH_LINES[1]
    diagonal = TARGET_POSITIONS_M[b - 1] - TARGET_POSITIONS_M[a - 1]
    assert np.degrees(np.arctan2(diagonal[2], np.linalg.norm(diagonal[:2]))) == pytest.approx(45.0, abs=1.0)
    without = hughes_2011_network_scenario(include_birdbath=False)
    assert without.n_readings == 118 and (without.visible & ~NETWORK.visible).sum() == 0


def test_exact_readings_give_back_every_parameter():
    result = network_test_covariance(NETWORK, TRUE)
    np.testing.assert_allclose(result.parameter_estimates[0].values, TRUE.values, atol=1e-6 * result.parameter_estimates[0].standard_uncertainties().max())
    assert np.abs(result.target_points_m - NETWORK.targets_m).max() < 1e-9


def test_recovery_over_many_noisy_seeds_is_statistically_consistent_with_the_fitted_uncertainties():
    """Each seed: simulate both-face readings from the Hughes-2011 uncompensated
    tracker with Table 4 noise, fit all 16 parameters from a rough start, and
    express the error of each in its own fitted standard uncertainty. If the
    estimates are unbiased and the covariance right, z is standard normal:
    about 68% within 1, 95% within 2, 99.7% within 3 (mu is excluded: its truth
    equals its prior mean, so it is trivially recovered)."""
    z = []
    for seed in range(40):
        fit = fit_network_test(NETWORK, simulate_network_test(NETWORK, TRUE, seed), seed)
        assert fit.success
        estimate = fit.parameter_estimates[0]
        z.append(((estimate.values - TRUE.values) / estimate.standard_uncertainties())[np.arange(N_PARAMETERS) != INDEX["mu"]])
    a = np.abs(np.array(z))
    assert 0.60 < (a < 1).mean() < 0.78 and 0.92 < (a < 2).mean() and (a < 3).mean() > 0.985
    assert 0.85 < np.sqrt((a**2).mean()) < 1.15


def test_fitted_uncertainties_are_the_same_order_as_the_papers_tables():
    """Not a match -- the layout is a reconstruction -- but every parameter within
    a factor of two of Tables 2-3, and the target uncertainties in the paper's
    2.1-4.9 um range to within a factor of two."""
    result = network_test_covariance(NETWORK, TRUE)
    ours = result.parameter_estimates[0].standard_uncertainties()
    paper = TRUE.standard_uncertainties()
    ratio = ours / paper
    assert ratio[INDEX["mu"]] == pytest.approx(1.0)
    assert np.all((ratio > 0.5) & (ratio < 1.25))
    targets = [np.sqrt(np.trace(result.target_covariance_m2[3 * i:3 * i + 3, 3 * i:3 * i + 3])) * 1e6 for i in range(15)]
    assert 1.0 < min(targets) and max(targets) < 9.8


def test_the_three_significant_correlations_the_paper_names_appear():
    """gamma-b_y, a_E0-b_z and a_E2-a_E0 are the paper's 'only significant
    correlations'. They come out of the reconstructed network too, with the
    signs below; no other pair among the geometric parameters reaches them."""
    c, _ = correlation(NETWORK)
    named = {("gamma_rad", "b_y0_m"): -1, ("a_E0_rad", "b_z0_m"): +1, ("a_E2_rad", "a_E0_rad"): -1}
    for (a, b), sign in named.items():
        assert sign * c[INDEX[a], INDEX[b]] > 0.5, (a, b, c[INDEX[a], INDEX[b]])


def test_raising_the_maximum_elevation_weakens_the_second_order_elevation_correlation():
    """The paper: a_E,2 and a_E,0 are correlated because no elevation exceeded
    57 degrees (mean cos 2phi = 0.69). With nests close to the zenith the
    correlation weakens, in the direction the paper describes."""
    base, _ = correlation(NETWORK)
    raised_network = hughes_2011_network_scenario(n_high_targets=2)
    raised, _ = correlation(raised_network)
    elevation = raised_network.ranges_and_elevations()[1][raised_network.visible]
    assert elevation.max() > 70.0
    i, j = INDEX["a_E2_rad"], INDEX["a_E0_rad"]
    assert abs(raised[i, j]) < abs(base[i, j]) - 0.03


@pytest.mark.parametrize("faces", [((FRONT, BACK),) * 5, ((FRONT,),) * 5])
def test_the_birdbath_lines_pin_down_the_range_offset(faces):
    """The paper adds the lines to separate the range offset lambda from the
    transit-axis offset e_x. In this reconstruction the *correlation coefficient*
    is moderate with and without them (about -0.3 and -0.2): e_x acts on range as
    e_x (1 - cos phi) on the front face and e_x (1 + cos phi) on the back face,
    so the elevation spread and the second face already separate it. What the
    lines do is tighten lambda: removing them raises its uncertainty by about a
    third, whereas removing the same number of other readings costs less than half
    as much (the control: about 3% with two faces, 10% with one)."""
    with_lines = replace(NETWORK, faces=faces)
    without = replace(hughes_2011_network_scenario(include_birdbath=False), faces=faces)
    c_with, s_with = correlation(with_lines)
    c_without, s_without = correlation(without)
    i, j = INDEX["lambda_m"], INDEX["e_x_m"]
    assert abs(c_with[i, j]) < 0.5 and abs(c_without[i, j]) < 0.5
    assert s_without[i] > 1.2 * s_with[i]
    rng = np.random.default_rng(0)
    pairs = [(s, t) for s in (3, 4) for t in range(15) if NETWORK.visible[s, t]
             and (s, t) not in {(3, 5), (3, 7), (4, 8), (4, 9)}]
    visible = NETWORK.visible.copy()
    for k in rng.choice(len(pairs), 4, replace=False):
        visible[pairs[k]] = False
    _, s_control = correlation(replace(NETWORK, faces=faces, visible=visible))
    assert (s_control[i] - s_with[i]) < 0.5 * (s_without[i] - s_with[i])  # the lines matter more than 4 other pairs
    if faces[0] == (FRONT,):
        assert s_without[j] > 1.5 * s_with[j]  # one face: the lines also pin down e_x
