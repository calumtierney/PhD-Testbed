"""What a review is told, and what it will not choose for you.

`ReviewSettings.tracker_model` has **no default**. Which instrument model the
risk is computed with (sensor noise alone, or the NPL geometric model in consider
mode) changes the numbers materially (docs/step1b_npl_geometric_model.md), and
which to use is a decision for whoever owns the review, not for this module. It is
printed on every table, figure and log record.
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from instruments.laser_tracker import LaserTracker
from instruments.npl_model import NPLErrorParameters

TRACKER_MODELS = ("noise_only", "npl_residual")
DECISION_RULES = ("simple", "best")
ELEVATION_LIMITS = ("none", "api_t3")


@dataclass(frozen=True)
class ReviewSettings:
    """Settings for one comparison. Every plan in it is scored with exactly these.

    tracker_model : REQUIRED. "noise_only" is the step-1 sensor-noise instrument
        (`LaserTracker()`); "npl_residual" is sensor noise plus the NPL geometric
        model's residual parameter errors (`NPLErrorParameters.hughes_2011_residual()`,
        one API T3 in IFM mode, 2011) in consider mode: errors shared by every
        station that uses the same physical tracker.
    systematic_floor_um : an isotropic placement-irreducible floor (refraction,
        thermal drift, nest repeatability, gravity, fixturing; `LaserTracker.
        systematic_std_m`), added once per characteristic. With "npl_residual" it is
        taken to cover only what the geometric model does not.
    decision_rule : "simple" (accept inside the tolerance limits) or "best" (accept
        when the part is more likely conforming than not; tabulating its risk takes
        minutes for process knowledge from few builds).
    n_builds : for the built-in factory cell only, the number of earlier parts its
        placeholder process statistics are read as coming from (None: known exactly).
        A QIF scene's clusters carry their own.
    require_all_characteristics : False (the default) prices a characteristic no
        station measures at its process prior and reports it; True makes that a
        blocking finding (the plan is infeasible). Applies to the plan being
        reviewed; the algorithm counterparts always measure everything.
    elevation_limits : "none" (the default; what `plan_qif_cell` assumes) or
        "api_t3": the tracker's mechanical elevation range, -60 to +77 degrees
        (Hughes et al. 2011 section 8; TODO(source): GKN's tracker).
    de_maxiter, de_popsize, de_tol, seed : the one optimiser (differential
        evolution over arc length and height per station) and its settings, the
        same for every objective in `algorithm_counterparts` (CLAUDE.md §3).
    """

    tracker_model: str
    systematic_floor_um: float = 0.0
    decision_rule: str = "simple"
    n_builds: Optional[int] = None
    require_all_characteristics: bool = False
    elevation_limits: str = "none"
    de_maxiter: int = 60
    de_popsize: int = 15
    de_tol: float = 1e-8
    seed: int = 0

    def __post_init__(self):
        if self.tracker_model not in TRACKER_MODELS:
            raise ValueError(f"tracker_model must be chosen from {list(TRACKER_MODELS)}, got {self.tracker_model!r}")
        if self.decision_rule not in DECISION_RULES:
            raise ValueError(f"decision_rule must be one of {list(DECISION_RULES)}, got {self.decision_rule!r}")
        if self.elevation_limits not in ELEVATION_LIMITS:
            raise ValueError(f"elevation_limits must be one of {list(ELEVATION_LIMITS)}, got {self.elevation_limits!r}")
        if not self.systematic_floor_um >= 0.0:
            raise ValueError("systematic_floor_um must be >= 0")
        if self.n_builds is not None and self.n_builds < 2:
            raise ValueError("n_builds must be at least 2 (a sample standard deviation needs two builds) or None")
        if self.de_maxiter < 1 or self.de_popsize < 2:
            raise ValueError("de_maxiter must be >= 1 and de_popsize >= 2")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewSettings":
        if "tracker_model" not in data:
            raise ValueError("tracker_model is required: there is no default (see ReviewSettings)")
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown setting(s) {unknown}")
        return cls(**data)

    def de_kwargs(self) -> Dict[str, Any]:
        return dict(maxiter=self.de_maxiter, popsize=self.de_popsize, tol=self.de_tol, workers=1)

    def build_tracker(self) -> LaserTracker:
        floor_m = self.systematic_floor_um * 1e-6
        if self.tracker_model == "noise_only":
            return LaserTracker(systematic_std_m=floor_m)
        return LaserTracker(systematic_std_m=floor_m, geometric_params=NPLErrorParameters.hughes_2011_residual(),
                            floor_excludes_geometric_parameters=floor_m > 0.0)

    def tracker_label(self) -> str:
        """One line for a table or figure caption."""
        base = ("sensor noise only" if self.tracker_model == "noise_only"
                else "sensor noise + NPL geometric parameters (Hughes 2011 residual), consider mode")
        return base + (f", floor {self.systematic_floor_um:g} um" if self.systematic_floor_um > 0 else "")
