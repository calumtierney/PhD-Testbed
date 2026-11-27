"""Why a plan cannot be scored, or what is wrong with it: findings, never repairs.

A plan an engineer made may be infeasible under the testbed's network rules: a station that
sees too few targets to be tied in, a part frame that cannot be fixed, a characteristic nobody
measures. This module says which, and for which station or characteristic. **It never repairs
anything**: it does not add a station, drop a target or substitute a shot. The plan is
reported as it is.

The rules are the ones `planning.cell.PartFrameScorer.score_visibility` applies, and
`tests/test_review_feasibility.py` pins the two together: a layout is feasible for the scorer if
and only if there is no blocking finding. They are:

  * every station but the first sees at least 3 targets (`MIN_TARGETS_PER_NON_ANCHOR_STATION`);
  * every station is tied to the first by a chain of stations sharing at least 3 targets
    pairwise (three shared, non-collinear targets fix a station's six degrees of freedom;
    `network.design.count_untied_stations` counts the exceptions, here they are named);
  * at least 3 of the part's datum targets are seen, and they are not collinear (otherwise
    the part frame cannot be fixed);
  * the network is rigid (the covariance solve is well conditioned);
  * every characteristic is seen: a blocking finding only if the review is told to require it,
    otherwise a warning, and the characteristic is priced at its process prior.

Stations are numbered from 1 in the order the plan lists them; station 1 is the one the others
are tied to.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from geometry.visibility import check_target_visibility
from network.design import datum_referenced_functional
from planning.objectives import MIN_TARGETS_PER_NON_ANCHOR_STATION

BLOCKING, WARNING = "blocking", "warning"


@dataclass(frozen=True)
class Finding:
    kind: str
    severity: str
    detail: str
    station: Optional[int] = None          # 1-based
    characteristic: Optional[str] = None
    target: Optional[str] = None

    def to_dict(self) -> Dict:
        return {"kind": self.kind, "severity": self.severity, "detail": self.detail, "station": self.station,
                "characteristic": self.characteristic, "target": self.target}


def untied_stations(visible: np.ndarray, anchor_index: int = 0, min_shared: int = MIN_TARGETS_PER_NON_ANCHOR_STATION) -> List[int]:
    """Indices (0-based) of the stations that cannot be attached to the anchor's frame: the stations
    `network.design.count_untied_stations` counts, named. Same procedure: grow the anchor's group of
    known targets, station by station, adding any station that shares at least `min_shared` of them."""
    seen = [set(np.nonzero(row)[0]) for row in np.asarray(visible, dtype=bool)]
    group = set(seen[anchor_index])
    remaining = [s for s in range(len(seen)) if s != anchor_index]
    changed = True
    while changed and remaining:
        changed = False
        for s in list(remaining):
            if len(seen[s] & group) >= min_shared:
                group |= seen[s]
                remaining.remove(s)
                changed = True
    return remaining


def diagnose(scenario, datum_indices: Sequence[int], characteristic_names: Sequence[str], visible: np.ndarray,
             require_all_characteristics: bool) -> List[Finding]:
    """The findings that need only the pattern of what each station measures (the network solve's own
    rigidity test is the scorer's, and is added by the caller when everything here passes)."""
    visible = np.asarray(visible, dtype=bool)
    findings: List[Finding] = []
    counts = visible.sum(axis=1)
    for s in range(1, visible.shape[0]):
        if counts[s] < MIN_TARGETS_PER_NON_ANCHOR_STATION:
            findings.append(Finding(
                "station_thin", BLOCKING, f"station {s + 1} measures {int(counts[s])} target(s); at least "
                f"{MIN_TARGETS_PER_NON_ANCHOR_STATION} are needed to tie it to the network", station=s + 1))
    for s in untied_stations(visible, 0):
        if counts[s] >= MIN_TARGETS_PER_NON_ANCHOR_STATION:  # thin stations are already named above
            findings.append(Finding(
                "station_untied", BLOCKING, f"station {s + 1} shares fewer than {MIN_TARGETS_PER_NON_ANCHOR_STATION} measured targets "
                "with the stations tied to station 1, so it cannot be tied to station 1's frame", station=s + 1))
    seen = visible.any(axis=0)
    datum = [i for i in datum_indices if seen[i]]
    if len(datum) < 3:
        findings.append(Finding(
            "datum_too_few", BLOCKING, f"{len(datum)} of the part's {len(datum_indices)} datum targets are measured; at least 3 are "
            "needed to fix the part frame"))
    else:
        points = scenario.target_points_m[datum]
        if datum_referenced_functional(points, 0, list(range(len(datum)))) is None:
            findings.append(Finding(
                "datum_not_rigid", BLOCKING, f"the {len(datum)} measured datum targets are in a line (or nearly), so they cannot fix "
                "all six degrees of freedom of the part frame"))
    for name, characteristic in zip(characteristic_names, scenario.characteristics):
        if not seen[characteristic.target_indices[0]]:
            findings.append(Finding(
                "characteristic_not_seen", BLOCKING if require_all_characteristics else WARNING,
                f"{name} is not measured by any station" + ("" if require_all_characteristics else
                                                           "; it is priced at its process prior, as accepted or rejected unseen"),
                characteristic=name))
    return findings


def listed_target_findings(poses, scene, scenario, listed: Dict[int, List[int]], visible_full: np.ndarray) -> List[Finding]:
    """For each station that lists the targets it measures, name every listed target the station cannot see
    and why (range, elevation, incidence, occlusion). Such a shot is not counted: it is reported, never
    dropped silently. `visible_full` is the (stations x targets) visibility before the listing is applied."""
    findings: List[Finding] = []
    normals = scenario.target_normals_m
    for s, indices in listed.items():
        for t in indices:
            if visible_full[s, t]:
                continue
            result = check_target_visibility(
                poses[s].position_m, scenario.target_points_m[t], scenario.visibility_limits,
                target_normal_m=None if normals is None else normals[t], obstacles=scenario.obstacles,
                instrument_orientation=poses[s].orientation)
            reason = result.reason.value if not result.is_visible else "not visible"
            findings.append(Finding(
                "listed_target_not_visible", WARNING,
                f"station {s + 1} lists {scene.target_names[t]!r}, which it cannot see here ({reason}); the shot is not counted",
                station=s + 1, target=scene.target_names[t]))
    return findings
