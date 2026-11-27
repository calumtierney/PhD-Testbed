"""Feasibility findings (review.feasibility) are pinned to the scorer's own rules: a layout is feasible for
`PartFrameScorer.score_visibility` if and only if the findings agree. They explain; they never repair."""
import numpy as np
import pytest

from geometry.pose import InstrumentPose
from network.design import count_untied_stations
from planning.candidates import levelled_orientation
from review.core import build_context
from review.feasibility import BLOCKING, diagnose, untied_stations
from review.scene import ReviewScene
from review.settings import ReviewSettings


@pytest.fixture(scope="module")
def context():
    return build_context(ReviewScene.from_factory_cell(), ReviewSettings("noise_only"))


def random_layouts(context, n, seed):
    rng = np.random.default_rng(seed)
    aim = context.scene.aim_point_m
    for _ in range(n):
        k = int(rng.integers(1, 6))
        positions = np.column_stack([rng.uniform(-6.0, 21.0, k), rng.uniform(-8.0, 8.0, k), rng.uniform(0.8, 3.0, k)])
        yield [InstrumentPose(p, levelled_orientation(p, aim)) for p in positions]


def test_untied_stations_names_exactly_the_stations_count_untied_stations_counts():
    rng = np.random.default_rng(0)
    for _ in range(200):
        visible = rng.random((int(rng.integers(1, 7)), 30)) < rng.uniform(0.05, 0.5)
        names = untied_stations(visible, 0)
        assert len(names) == count_untied_stations(visible, 0, 3)
        assert 0 not in names


@pytest.mark.parametrize("require_all", [False, True])
def test_blocking_findings_agree_with_the_scorer_on_random_layouts(context, require_all):
    scorer_context = build_context(context.scene, ReviewSettings("noise_only", require_all_characteristics=require_all))
    scene, scorer = scorer_context.scene, scorer_context.scorer
    n_feasible = n_infeasible = n_not_rigid = 0
    for poses in random_layouts(scorer_context, 120, seed=3):
        visible = scorer.visible(poses)
        score = scorer.score_visibility(poses, visible)
        findings = diagnose(scorer_context.scenario, scene.datum_indices, scene.characteristic_names, visible, require_all)
        blocking = [f for f in findings if f.severity == BLOCKING]
        if score.feasible:
            n_feasible += 1
            assert not blocking, [f.detail for f in blocking]
        elif score.deficit > 0.0:
            n_infeasible += 1
            assert blocking, "the scorer says infeasible (deficit > 0) but no reason was found"
        else:
            n_not_rigid += 1  # singular network: `review_with_context` adds network_not_rigid
            assert not blocking
    assert n_feasible >= 5 and n_infeasible >= 5, (n_feasible, n_infeasible, n_not_rigid)


def test_every_finding_names_its_station_or_characteristic(context):
    scene, scorer = context.scene, context.scorer
    for poses in random_layouts(context, 60, seed=5):
        findings = diagnose(context.scenario, scene.datum_indices, scene.characteristic_names, scorer.visible(poses), True)
        for f in findings:
            if f.kind in ("station_thin", "station_untied"):
                assert f.station is not None and 1 <= f.station <= len(poses) and f"station {f.station}" in f.detail
            if f.kind == "characteristic_not_seen":
                assert f.characteristic in scene.characteristic_names
