"""Tests for planning.cell: the factory cell with a reference network."""
import json
from pathlib import Path

import numpy as np
import pytest

from planning.cell import (
    CellLayoutScorer, CellRiskModel, MaxUncertaintyObjective, RiskObjective, TraceObjective, build_risk_curve,
    ers_nests, factory_cell_scenario, layout_poses, measurement_time_minutes, process_knowledge_for,
    search_cell_layout, standard_objectives,
)
from planning.experiment import long_part_track
from planning.objectives import evaluate_plan, objective_a0_trace_covariance, objective_c_direct_risk, station_visibility_matrix
from risk.decision import ProcessKnowledge, decision_risk, unmeasured_risk

TRACK = long_part_track()
ROW_PLUS_Y = TRACK.row_length_m + np.pi * TRACK.loop_radius_m  # s at the +x end of the +y row
LOOPS_4 = [(15.9, 1.28), (27.5, 1.45), (45.0, 1.17), (54.4, 1.18)]  # the earlier 4-station plan, all on end loops
ROWS_2 = [(11.0, 1.25), (ROW_PLUS_Y + 11.0, 1.25)]  # one station on each row, in front of each cluster


@pytest.fixture(scope="module")
def cell():
    return factory_cell_scenario(track=TRACK)


@pytest.fixture(scope="module")
def bare():
    return factory_cell_scenario(track=TRACK, with_reference_network=False)


def test_reference_nests_face_into_the_cell_and_keep_existing_targets_in_place(cell, bare):
    points, normals = ers_nests()
    assert len(points) == 28 and np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    centre = np.array([7.5, 0.0, 0.0])
    inward = np.einsum("ij,ij->i", normals[:, :2], (centre - points)[:, :2])
    assert np.all(inward > 0)  # every nest faces the cell
    assert len(cell.target_points_m) == len(bare.target_points_m) + 28
    assert np.array_equal(cell.target_points_m[: len(bare.target_points_m)], bare.target_points_m)
    assert [c.name for c in cell.characteristics] == [c.name for c in bare.characteristics]
    # nests stand outside the track: no station can be nearer the part than a nest's pillar line
    assert np.all((np.abs(points[:, 1]) > TRACK.y_left_row_m) | (points[:, 0] < -TRACK.loop_radius_m) | (points[:, 0] > 15 + TRACK.loop_radius_m))


def test_scorer_reproduces_the_full_network_solve_and_the_scalar_visibility(bare):
    poses = layout_poses(LOOPS_4, bare, TRACK)
    scorer = CellLayoutScorer(bare)
    score = scorer.score(poses)
    reference = evaluate_plan(poses, bare)
    assert score.feasible
    assert np.allclose(score.u_m, list(reference.per_characteristic_uncertainty_m.values()), rtol=1e-6)
    assert np.sum(score.trace_m2) == pytest.approx(objective_a0_trace_covariance(reference, bare), rel=1e-6)
    assert np.array_equal(scorer.visible(poses), station_visibility_matrix(bare, poses))


def test_the_reference_network_lets_row_stations_tie_in_and_cuts_uncertainty(cell, bare):
    """Physical gate: without the network two row stations cannot be tied to the anchor at all;
    with it they can, and the same four loop stations see every characteristic 3-4 times better."""
    assert not CellLayoutScorer(bare).score(layout_poses(ROWS_2, bare, TRACK)).feasible
    rows = CellLayoutScorer(cell).score(layout_poses(ROWS_2, cell, TRACK))
    assert rows.feasible and np.all(rows.u_m < 15e-6)
    before = CellLayoutScorer(bare).score(layout_poses(LOOPS_4, bare, TRACK))
    after = CellLayoutScorer(cell).score(layout_poses(LOOPS_4, cell, TRACK))
    assert np.all(after.u_m < before.u_m / 2.5)  # more shared targets: better-determined station poses
    assert after.n_shots > before.n_shots


def test_one_station_cannot_see_both_faces_unless_characteristics_may_go_unobserved(cell):
    one = layout_poses([(11.0, 1.25)], cell, TRACK)  # in front of the marginal cluster only
    strict = CellLayoutScorer(cell).score(one)
    assert not strict.feasible and strict.deficit == 3.0  # the three characteristics on the far face
    relaxed = CellLayoutScorer(cell, require_all_characteristics=False).score(one)
    assert relaxed.feasible and np.isnan(relaxed.u_m[:3]).all() and np.isfinite(relaxed.u_m[3:]).all()


def test_risk_curves_match_direct_evaluation_and_respect_the_unmeasured_ceiling():
    prior = ProcessKnowledge.from_builds(38e-6, 3e-6, 5)
    simple = build_risk_curve(prior, -50e-6, 50e-6, "simple")
    best = build_risk_curve(prior, -50e-6, 50e-6, "best")
    for u in (2e-6, 6.3e-6, 17e-6):
        assert simple(u) == pytest.approx(sum(decision_risk(prior, -50e-6, 50e-6, u)), rel=0.02)
        assert best(u) <= simple(u) * 1.001
        assert best(u) <= unmeasured_risk(prior, -50e-6, 50e-6) * 1.001
    assert best(float("nan")) == unmeasured_risk(prior, -50e-6, 50e-6) == simple(float("nan"))
    assert best(2e-6) < best(6.3e-6) < best(17e-6)
    with pytest.raises(ValueError):
        build_risk_curve(prior, -50e-6, 50e-6, "nonsense")


def test_known_simple_risk_model_is_objective_c(bare):
    poses = layout_poses(LOOPS_4, bare, TRACK)
    score = CellLayoutScorer(bare).score(poses)
    model = CellRiskModel.build(bare, None, "simple")
    assert model.total(score.u_m) == pytest.approx(objective_c_direct_risk(evaluate_plan(poses, bare), bare), rel=0.02)
    objectives = standard_objectives(bare)
    assert set(objectives) == {"A0", "A1", "B", "C"}
    assert objectives["A1"](score) == pytest.approx(np.sum(score.u_m))
    assert objectives["C"](score) == pytest.approx(model.total(score.u_m))
    assert MaxUncertaintyObjective()(score) == np.max(score.u_m)


def test_objectives_refuse_a_layout_that_leaves_a_characteristic_unobserved_but_risk_prices_it(cell):
    one = layout_poses([(11.0, 1.25)], cell, TRACK)
    score = CellLayoutScorer(cell, require_all_characteristics=False).score(one)
    assert TraceObjective()(score) == float("inf") and MaxUncertaintyObjective()(score) == float("inf")
    model = CellRiskModel.build(cell, 5, "best")
    priced = RiskObjective(model)(score)
    assert priced < model.unmeasured_total()  # measuring three of six helps
    assert priced > 3 * model.curves[0].unmeasured * 0.999  # the three unobserved ones cost their unmeasured risk


def test_assignment_doubt_is_a_mixture_with_the_other_cluster(cell):
    plain = process_knowledge_for(cell, "cluster1_profile_0", None)
    doubted = process_knowledge_for(cell, "cluster1_profile_0", None, propagate_assignment_doubt=True)
    assert len(plain.components) == 1 and len(doubted.components) == 2
    marginal = process_knowledge_for(cell, "cluster2_profile_0", None)
    expected = 0.9 * plain.nonconformance(-50e-6, 50e-6) + 0.1 * marginal.nonconformance(-50e-6, 50e-6)
    assert doubted.nonconformance(-50e-6, 50e-6) == pytest.approx(expected)


def test_search_returns_a_feasible_layout_in_track_coordinates(cell):
    scorer = CellLayoutScorer(cell)
    result = search_cell_layout(scorer, standard_objectives(cell)["A1"], TRACK, 2, seed=0, maxiter=25, popsize=10, tol=1e-8, workers=1)
    assert result.value is not None and len(result.stations) == 2
    assert all(seg in ("-y row", "+x loop", "+y row", "-x loop") for _, _, seg in result.stations)
    rescored = scorer.score(layout_poses(result.stations, cell, TRACK))
    assert np.sum(rescored.u_m) == pytest.approx(result.value)
    assert result.value <= np.sum(scorer.score(layout_poses(ROWS_2, cell, TRACK)).u_m) * 1.5  # in the same league as a sensible hand layout


def test_time_model_reproduces_its_one_data_point():
    times = measurement_time_minutes(5, 126)
    assert times["per_station_model_min"] == pytest.approx(55.0) and times["per_observation_model_min"] == pytest.approx(55.0)
    assert measurement_time_minutes(3, 126)["per_station_model_min"] == pytest.approx(33.0)
