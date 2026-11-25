"""The network test of Hughes et al. (2011) as a scenario: 15 target nests, 5
tracker positions, both faces, fitted for the tracker's 16 geometric parameters
(`network.solve` estimate mode).

**Every coordinate here is a reconstruction, not a transcription.** The paper
gives a description (section 6, figures 4-6) and a few numbers; it does not give
coordinates. What is taken from it, and checked by tests:

  * 15 target holders: 8 on two walls meeting at a corner, 1 on the floor, 6 on
    a bench, at different heights;
  * 5 tracker positions: 0 on the wheeled tripod, which sets the coordinate
    frame; 1 at the same place as 0 but rotated; 2 on the stand raised to full
    height; 3 on the bench; 4 on a floor trivet;
  * ranges 0.27-6.4 m and elevations no higher than 57 degrees (the tracker's
    mechanical range is -60 to +77);
  * two "birdbath" lines of sight to break the lambda-e_x correlation: targets 6
    and 8 viewed from positions 3 and 0 (horizontal), and 9 and 10 viewed from
    0 and 4 (a vertical diagonal about 45 degrees), the tracker outside
    (external) and between (internal) the pair;
  * every target measured from every position except those with an obscured line
    of sight, on both faces: 126 readings out of a possible 15 x 5 x 2 = 150, i.e.
    63 of the 75 target-position pairs.

Which 12 pairs are obscured is not stated (`OBSCURED_PAIRS`, TODO(source)).
The layout is the author's; the room is 6.4 m by 3 m by 3 m with the walls at
x = 0 (left) and y = 3 (back), meeting at the corner (0, 3). None of it was
tuned to improve the fitted uncertainties against the paper's tables: the
comparison in `viz/run_network_test_reproduction.py` is what the layout gives.
Tracker positions are levelled (standing axis vertical); only the yaw differs.
"""
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker
from instruments.npl_model import BACK, FRONT, NPLErrorParameters
from network.solve import NetworkSolveResult, Observation, ParameterPrior, perturbed_initial_guess, simulate_observations, solve_network

# stations: (position, yaw in radians) -- TODO(source): all of it
STATION_POSITIONS_M = np.array([
    [4.4, 2.0, 1.4],    # 0 tripod at mid height; the frame; on both birdbath lines
    [4.4, 2.0, 1.4],    # 1 the same place, rotated
    [0.27, 1.2, 2.0],   # 2 stand at full height, near the left wall
    [2.6, 2.0, 1.4],    # 3 on the bench: the external point of the horizontal birdbath line
    [4.4, 1.1, 0.5],    # 4 floor trivet: the external point of the diagonal birdbath line
])
STATION_YAW_RAD = np.array([0.0, 1.3, -0.6, 2.4, 3.6])

# targets 1..15 -- TODO(source)
TARGET_POSITIONS_M = np.array([
    [0.0, 0.6, 1.3], [0.0, 1.2, 2.0], [0.0, 2.2, 2.6],           # 1-3  left wall
    [3.0, 1.2, 1.2], [3.2, 2.6, 1.1], [3.4, 2.0, 1.4],           # 4-6  bench
    [5.0, 2.6, 1.0], [5.4, 2.0, 1.4], [4.4, 1.4, 0.8],           # 7-9  bench (6 and 8 horizontal with 3 and 0; 9 on the diagonal)
    [4.4, 3.0, 2.4], [2.0, 3.0, 1.6], [5.8, 3.0, 0.6],           # 10-12 back wall
    [5.0, 0.6, 0.0],                                              # 13   floor
    [6.4, 3.0, 2.3], [2.3, 3.0, 2.98],                            # 14-15 back wall
])
WALL_TARGETS = (1, 2, 3, 10, 11, 12, 14, 15)
BENCH_TARGETS = (4, 5, 6, 7, 8, 9)
FLOOR_TARGETS = (13,)

# (station, target) pairs with an obscured line of sight, 1-based targets: 12 of 75 -- TODO(source)
OBSCURED_PAIRS = ((2, 9), (2, 13), (2, 7), (2, 12), (3, 13), (3, 12), (3, 1), (3, 4), (4, 3), (4, 15), (4, 11), (4, 2))

# the birdbath lines: (targets, external station, internal station)
BIRDBATH_LINES = (((6, 8), 3, 0), ((9, 10), 4, 0))

# extra targets near the zenith, for the variant that raises the maximum elevation
HIGH_TARGETS_M = np.array([[4.3, 2.4, 2.9], [0.7, 1.4, 2.9]])


@dataclass(frozen=True)
class HughesNetwork:
    """A network-test layout: `visible` (S, N) marks the target-position pairs that
    are measured (each on both faces)."""

    targets_m: np.ndarray
    station_poses: Tuple[InstrumentPose, ...]
    visible: np.ndarray
    faces: Tuple[Tuple[str, ...], ...]
    instrument_ids: np.ndarray

    @property
    def n_readings(self) -> int:
        return int(self.visible.sum()) * len(self.faces[0])

    def ranges_and_elevations(self) -> Tuple[np.ndarray, np.ndarray]:
        """(S, N) range (m) and elevation (degrees) of every pair, in each station's levelled frame."""
        positions = np.array([p.position_m for p in self.station_poses])
        delta = self.targets_m[None, :, :] - positions[:, None, :]
        ranges = np.linalg.norm(delta, axis=2)
        return ranges, np.degrees(np.arcsin(delta[:, :, 2] / ranges))

    def shots(self) -> List[Tuple[int, int]]:
        return [(int(s), int(t)) for s, t in zip(*np.nonzero(self.visible))]


def _yaw(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def hughes_2011_network_scenario(include_birdbath: bool = True, n_high_targets: int = 0) -> HughesNetwork:
    """The reconstructed network (see the module docstring).

    `include_birdbath=False` removes the readings that make the two birdbath lines:
    targets 6 and 8 from position 3, and 9 and 10 from position 4, so neither pair
    is seen from both the outside and the inside of its line. `n_high_targets` (0-2)
    adds nests close to the zenith above the trackers, raising the maximum
    elevation from 57 to about 73 degrees: the variant for the paper's remark that
    the second-order elevation parameters are correlated because no elevation
    exceeded 57 degrees.
    """
    targets = TARGET_POSITIONS_M.copy()
    if n_high_targets:
        targets = np.vstack([targets, HIGH_TARGETS_M[:n_high_targets]])
    visible = np.ones((len(STATION_POSITIONS_M), len(targets)), bool)
    for station, target in OBSCURED_PAIRS:
        visible[station, target - 1] = False
    if not include_birdbath:
        for (a, b), external, _ in BIRDBATH_LINES:
            visible[external, a - 1] = visible[external, b - 1] = False
    poses = tuple(InstrumentPose(position_m=p.copy(), orientation=_yaw(yaw)) for p, yaw in zip(STATION_POSITIONS_M, STATION_YAW_RAD))
    n = len(poses)
    return HughesNetwork(targets, poses, visible, ((FRONT, BACK),) * n, np.zeros(n, int))


def simulate_network_test(network: HughesNetwork, true_params: NPLErrorParameters, seed: int,
                          tracker: Optional[LaserTracker] = None) -> List[Observation]:
    """Both-face readings of the network from a tracker whose real parameters are
    `true_params`, with the tracker's random noise (Table 4 by default)."""
    return simulate_observations(network.targets_m, list(network.station_poses), tracker or LaserTracker(), np.random.default_rng(seed),
                                 instrument_params=[true_params], instrument_ids=network.instrument_ids, faces=network.faces,
                                 visible=network.visible)


# the paper's prior information: only the range scale, traceable to the laser wavelength (Table 2: 5e-7)
NETWORK_TEST_PRIOR = ParameterPrior.flat_except({"mu": 5e-7})


def fit_network_test(network: HughesNetwork, observations: Sequence[Observation], seed: int = 0,
                     prior: ParameterPrior = NETWORK_TEST_PRIOR, tracker: Optional[LaserTracker] = None,
                     start_position_error_m: float = 2e-3, start_rotation_error_rad: float = 2e-3) -> NetworkSolveResult:
    """Fit targets, station poses (position 0 fixed) and the 16 parameters, from a
    rough start (the truth perturbed by `start_*` -- what a surveyed layout would
    give) with the parameters at their prior mean."""
    targets, poses = perturbed_initial_guess(network.targets_m, list(network.station_poses), start_position_error_m,
                                             start_rotation_error_rad, np.random.default_rng(10_000 + seed), anchor_index=0)
    return solve_network(list(observations), targets, poses, tracker or LaserTracker(), anchor_index=0,
                         instrument_ids=network.instrument_ids, estimate_parameters=prior)


def network_test_covariance(network: HughesNetwork, true_params: NPLErrorParameters, prior: ParameterPrior = NETWORK_TEST_PRIOR,
                            tracker: Optional[LaserTracker] = None) -> NetworkSolveResult:
    """The covariance the network test delivers, with no random noise: exact
    readings of the layout, fitted from the truth. The posterior covariance of
    the fit depends on the geometry and the noise levels, not on the data."""
    tracker = tracker or LaserTracker()
    exact = LaserTracker(sigma_d_m=1e-30, sigma_theta_rad=1e-30, sigma_phi_rad=1e-30)
    observations = simulate_network_test(network, true_params, 0, exact)
    return solve_network(observations, network.targets_m, list(network.station_poses), tracker, anchor_index=0,
                         instrument_ids=network.instrument_ids, estimate_parameters=prior)
