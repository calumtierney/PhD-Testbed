"""The planning scenarios' default instrument: sensor noise plus the NPL geometric model's residual
parameter errors, in consider mode (`instruments.laser_tracker.default_planning_tracker`).

`LaserTracker()` stays the step-1 sensor-noise-only instrument (what the CLAUDE.md §4b gate tests);
a scenario builder given no tracker uses the planning default, and one given a tracker uses that.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from instruments.laser_tracker import LaserTracker, default_planning_tracker
from instruments.npl_model import NPLErrorParameters
from planning.experiment import (
    default_two_cluster_scenario, long_part_track, long_part_two_cluster_scenario, mixed_tolerance_two_cluster_scenario,
    three_cluster_scenario, wing_four_point_scenario,
)
from planning.objectives import evaluate_plan
from planning.cell import factory_cell_scenario
from planning.inspection import build_inspection_scenario
from planning.candidates import hemisphere_candidate_poses

SAMPLE = Path(__file__).resolve().parent.parent / "scenes" / "qif" / "SheetMetal_QIF_Results_6_samples.QIF"

BUILDERS = [
    ("headline", lambda **kw: default_two_cluster_scenario(**kw)),
    ("three cluster", lambda **kw: three_cluster_scenario(**kw)),
    ("mixed tolerance", lambda **kw: mixed_tolerance_two_cluster_scenario(**kw)),
    ("wing four point", lambda **kw: wing_four_point_scenario(**kw)),
    ("long part on a track", lambda **kw: long_part_two_cluster_scenario(**kw)),
]


def test_the_planning_default_is_the_residual_parameter_set_in_consider_mode_and_warns_of_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a double-counting warning would be a bug in the default itself
        tracker = default_planning_tracker()
    residual = NPLErrorParameters.hughes_2011_residual()
    assert tracker.geometric_params == residual
    assert tracker.geometric_params.covariance.any() and not tracker.geometric_params.values.any()  # zero mean
    assert tracker.systematic_std_m == 0.0
    noise_only = LaserTracker()
    assert noise_only.geometric_params is None  # the step-1 instrument is unchanged
    assert (tracker.sigma_d_m, tracker.sigma_theta_rad, tracker.sigma_phi_rad) == (
        noise_only.sigma_d_m, noise_only.sigma_theta_rad, noise_only.sigma_phi_rad)  # same sensor noise


@pytest.mark.parametrize("name, build", BUILDERS)
def test_scenario_builders_default_to_the_planning_tracker_and_honour_an_explicit_one(name, build):
    assert build().tracker == default_planning_tracker()
    explicit = LaserTracker(systematic_std_m=3e-6)
    assert build(tracker=explicit).tracker is explicit


def test_the_factory_cell_and_the_qif_cell_use_the_planning_default():
    assert factory_cell_scenario(track=long_part_track()).tracker == default_planning_tracker()
    from interchange.qif import read_qif
    from planning.qif_cell import build_qif_cell
    cell = build_qif_cell(read_qif(SAMPLE))
    assert cell.scenario.tracker == default_planning_tracker()
    explicit = LaserTracker()
    assert build_qif_cell(read_qif(SAMPLE), tracker=explicit).scenario.tracker is explicit


def test_the_wing_box_inspection_experiment_is_still_noise_only():
    """Not switched: it is a separate experiment (its own scenario class and results), not among
    the scenes the side-by-side measured. Say so in a test so a later change is deliberate."""
    assert build_inspection_scenario().tracker == LaserTracker()


def test_parameter_errors_only_add_uncertainty_to_a_plan():
    scenario = default_two_cluster_scenario()
    noise_only = default_two_cluster_scenario(tracker=LaserTracker())
    poses = hemisphere_candidate_poses(scenario.target_points_m.mean(axis=0), 4.0, [0, 180], [20.0])
    full = evaluate_plan(poses, scenario)
    plain = evaluate_plan(poses, noise_only)
    assert full is not None and plain is not None
    for name, u_plain in plain.per_characteristic_uncertainty_m.items():
        assert full.per_characteristic_uncertainty_m[name] > u_plain  # strictly: the parameters add variance
    # and the cost of ignoring them is not small on this scene (CLAUDE.md §4d: angular share ~80%)
    ratio = np.array(list(full.per_characteristic_uncertainty_m.values())) / np.array(list(plain.per_characteristic_uncertainty_m.values()))
    assert 1.1 < ratio.min() and ratio.max() < 3.0
