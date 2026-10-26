"""Plumbing tests for planning.inspection_experiment on a deliberately tiny
budget. They check structure and the relationships the experiment depends on
(R-match's limits are exactly U's realised risk; a re-scored run keeps its
layouts; the seed set is shared), not search quality -- that is what the
experiment itself reports."""
import numpy as np
import pytest

from planning.inspection import build_candidate_pool, build_inspection_scenario, evaluate_layout
from planning.inspection_experiment import (
    cross_evaluate, rescore_run, run_inspection_experiment, seed_layouts_for,
)

TINY_DE = dict(maxiter=3, popsize=6, tol=1e-8, workers=1)


@pytest.fixture(scope="module")
def scenario():
    return build_inspection_scenario()


@pytest.fixture(scope="module")
def pool(scenario):
    return build_candidate_pool(scenario, n_candidates=600, seed=3)


@pytest.fixture(scope="module")
def result(scenario, pool):
    return run_inspection_experiment(scenario, k_values_u=range(4, 7), k_values_r=range(1, 4), pool=pool,
                                     de_kwargs=TINY_DE, backward_pass=False)


def test_seeds_cover_the_part_and_every_plan_gets_the_same_coverage_seeds(scenario, pool):
    seeds = seed_layouts_for(scenario, pool, n_each=2, max_stations=8)
    assert len(seeds) == 8 and all(len(s) >= 1 and s.shape[1] == 3 for s in seeds)
    covered = np.zeros(scenario.n_points, bool)
    for s in seeds:
        covered |= evaluate_layout(scenario, s, allow_unmeasured=True).visible.any(axis=0)
    assert covered.mean() > 0.95  # between them the seeds see (nearly) every point


def test_experiment_runs_and_reports_each_plan_with_ascending_sweeps(result):
    assert {"U", "R"} <= set(result.runs)
    for run in result.runs.values():
        ks = [r.n_stations for r in run.sweep]
        assert ks == sorted(ks) and len(set(ks)) == len(ks)
    assert result.u_curve and all(v > 0 for v in result.u_curve.values())
    assert set(result.knee_by_gain) <= {0.2, 0.1, 0.05, 0.02}


def test_r_match_is_held_to_exactly_the_risk_plan_u_delivers(result):
    if "R-match" not in result.runs:
        pytest.skip("tiny budget found no feasible plan U layout")
    chosen = result.runs["U"].answer
    spec = result.runs["R-match"].spec
    assert spec.max_p_escape == pytest.approx(chosen.outcome.p_escape)
    assert spec.max_e_fr == pytest.approx(chosen.outcome.e_fr)
    assert spec.allow_unmeasured
    assert chosen.outcome.measured.all()  # plan U measures every point


def test_a_rescored_run_keeps_its_layouts_and_changes_only_the_risk(result):
    other = build_inspection_scenario(tolerance_half_m=250e-6)
    again = rescore_run(other, result.runs["U"])
    for before, after in zip(result.runs["U"].sweep, again.sweep):
        assert np.array_equal(before.positions_m, after.positions_m)
        if before.outcome.feasible:
            assert np.allclose(before.outcome.u_m, after.outcome.u_m, equal_nan=True)  # geometry unchanged
            assert after.outcome.e_fr < before.outcome.e_fr  # a looser tolerance: fewer false rejections
    assert (again.answer is None) == (result.runs["U"].answer is None)


def test_cross_evaluate_reports_both_yardsticks(scenario, result):
    answer = result.runs["R"].sweep[-1]
    if not answer.outcome.feasible:
        pytest.skip("tiny budget found no feasible plan R layout")
    row = cross_evaluate(scenario, answer.positions_m)
    assert row["feasible"] and row["n_measured"] + row["n_unmeasured"] == scenario.n_points
    assert row["p_escape"] == pytest.approx(answer.outcome.p_escape)
    assert row["n_over_tur_limit"] >= 0 and row["max_point_pfa"] <= max(scenario.nonconformance)
