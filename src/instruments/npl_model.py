"""The NPL geometric error model of a laser tracker (Hughes et al. 2011, Meas.
Sci. Technol. 22 045103; docs/step1b_npl_geometric_model.md).

**What this adds to the chain.** `instruments.laser_tracker` uses only the
paper's *noise* estimates (Table 4): independent random errors on range and
the two angles. The paper's main subject is different: the *systematic*
errors of the instrument's mechanism -- axes that are offset or tilted, a
beam that does not leave the gimbal exactly where it should, encoders whose
scale is not perfectly uniform. After a tracker has been compensated those
errors are not gone, only known to a few micrometres and arcseconds, and
that residual is *one unknown per physical tracker*. It is shared by every
shot that tracker takes, at every station it is moved to, so it makes the
errors of different targets correlated -- which an independent per-point
noise term cannot represent. This module is the forward model: given the
parameters of one physical instrument it predicts what that instrument
*reads* (d, theta, phi) for a target at a true position, and, going the
other way, what true position a set of readings corresponds to. The network
layers (`network.solve`, `network.design`) use it to carry the parameter
uncertainty through to the target covariance.

**The mechanism, in metrology terms** (the paper's figures 1-3; local frame
exactly as `geometry.spherical`: standing axis = z, azimuth theta from +x
towards +y, elevation phi above the horizontal):

  * The beam is steered by two rotations: the head turns about the standing
    axis (azimuth theta), and the beam is tipped about the transit axis
    (elevation phi). A perfect tracker has the transit axis horizontal and
    through the origin, and the beam leaving from the origin at right angles
    to it. With every parameter zero the beam direction is exactly
    (cos phi cos theta, cos phi sin theta, sin phi) -- a test pins this.
  * `e_x`  the transit axis is offset from the standing axis (m);
    `b_y0, b_z0`  the beam leaves the gimbal offset sideways / vertically (m);
    `alpha`  the transit axis is tilted in the yz-plane (rad);
    `gamma`  the beam is not at right angles to the transit axis (rad).
    These are the offsets and misalignments a theodolite user knows as the
    transit-axis offset, trunnion-axis error and collimation error.
  * The angle encoders are not perfectly uniform: the reading differs from
    the true angle by a smooth function of the angle itself, written as a
    short Fourier series (first and second harmonic: eccentricity and scale
    error). `a_A,q`, `b_A,q` for azimuth; `a_E,q`, `b_E,q` for elevation;
    `a_E,0` is a constant offset of the elevation zero.
  * Range: `lambda` an offset (m), `mu` a scale correction: d = (1+mu) d* + lambda.

**The forward model** (the paper's equations 3-5, 8-10). The target is at

    x = b(theta*, phi*) + d* n_B(theta*, phi*),

with b the point the beam leaves from and n_B its direction, both after
rotating by the true azimuth theta* and elevation phi* (`beam_geometry`).
Note d* is measured from the beam source b, not from the origin. The
readings are

    d     = (1 + mu) d* + lambda
    theta = theta* + sum_q [a_A,q cos(q theta*) + b_A,q sin(q theta*)]
    phi   = phi*  + a_E,0 + sum_q [a_E,q cos(q phi*) + b_E,q sin(q phi*)]

**One convention differs from the paper, deliberately.** The paper writes
the Fourier terms at the *true* angles in its equations 4-5 but at the
*observed* angles in the correction equations 13 and 16-17. They differ by
a term of order (encoder error)^2 -- arcseconds of an arcsecond -- a few
nanometres at 6 m, a thousandth of the uncertainty of the quantities they
are used to estimate (tests quantify it). The forward model here uses the
true angles throughout, so `observed_to_true` is its exact inverse and a
round trip returns the point to rounding error; `convention="paper"`
reproduces the paper's correction for comparison.

**Two faces.** Equation 10 has two solutions: the front face (theta*, phi*)
and the back face, near (theta* + pi, pi - phi*), reached by turning the
head half a revolution and steering the beam over the top
(`geometry.spherical.cartesian_to_spherical_back_face`). A perfect tracker
gives the same point from both; a real one does not, and that is how the
parameters are separated. See `two_face_*` laws in
`tests/test_npl_model.py` and `viz/make_two_face_figure.py`.

**Units.** SI internally: metres and radians; mu is dimensionless. The
parameter vector `values` has the 16 entries of `PARAMETER_NAMES` in the
order of the paper's vector c = (lambda, mu, h): lambda, mu, e_x, b_y0,
b_z0, alpha, gamma, a_A1, b_A1, a_A2, b_A2, a_E0, a_E1, b_E1, a_E2, b_E2.
a_A,0 = 0 fixes the frame and b_x,0 = 0 because it cannot be told apart
from lambda (the paper's constraints), so they are not parameters.

**What is a placeholder.** Nothing here is a placeholder except mu: the
paper prints 1.000, which can only mean 1 + mu (mu ~ 0); TODO(source).
`hughes_2011_*` are one physical instrument -- an API T3, in IFM mode -- not
a typical tracker; fill an `NPLErrorParameters` from GKN's own network test
with `from_fit` / `from_json` instead of using them.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from geometry.spherical import cartesian_to_spherical, cartesian_to_spherical_back_face

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

FRONT = "front"
BACK = "back"

PARAMETER_NAMES = (
    "lambda_m", "mu",
    "e_x_m", "b_y0_m", "b_z0_m", "alpha_rad", "gamma_rad",
    "a_A1_rad", "b_A1_rad", "a_A2_rad", "b_A2_rad",
    "a_E0_rad", "a_E1_rad", "b_E1_rad", "a_E2_rad", "b_E2_rad",
)
N_PARAMETERS = len(PARAMETER_NAMES)
INDEX: Dict[str, int] = {name: i for i, name in enumerate(PARAMETER_NAMES)}

# Convergence of the iterative solve for the true coordinates (see
# `solve_true_coordinates`): stop when the beam point matches the target to
# 0.1 picometre -- far below double-precision position resolution at the
# ranges used, so the round trip is limited by rounding, not by this.
SOLVE_TOLERANCE_M = 1e-13
SOLVE_MAX_ITERATIONS = 25

# Finite-difference steps for the sensitivity of the readings to each
# parameter (`observation_parameter_jacobian`). The readings are smooth and
# very nearly linear in every parameter over its plausible range (the largest
# is alpha, 17 arcsec), so a central difference over a step of the size the
# parameter's own uncertainty is quoted in -- 1 um for lengths, 1e-6 for mu,
# 1 arcsec for angles -- cancels the curvature, is far above rounding noise
# (eps * |reading| / step ~ 1e-10) and far below any nonlinearity. Halving
# the step changes the result by less than 1e-6 relative (tests).
PARAMETER_STEPS = np.array(
    [1e-6, 1e-6] + [1e-6] * 3 + [ARCSEC_TO_RAD] * 11
)


# -- the parameter set -----------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class NPLErrorParameters:
    """The 16 parameters of *one physical tracker* and their covariance.

    `values` (16,): see the module docstring for names, order and units.
    `covariance` (16, 16): the covariance V_c of the parameter *errors* --
    what is not known about them. For a tracker compensated with the
    parameters' best estimates this is the covariance of those estimates, and
    `values` is the *remaining* error, zero on average (`hughes_2011_residual`).
    For a tracker running uncompensated `values` is the error itself
    (`hughes_2011_uncompensated`).

    Immutable, arrays read-only; equal if bytes are equal, so a
    `LaserTracker` holding one still compares and hashes.
    """

    values: np.ndarray
    covariance: np.ndarray
    label: str = ""

    def __post_init__(self):
        values = np.array(self.values, dtype=float)
        covariance = np.array(self.covariance, dtype=float)
        if values.shape != (N_PARAMETERS,):
            raise ValueError(f"values must have shape ({N_PARAMETERS},), got {values.shape}")
        if covariance.shape != (N_PARAMETERS, N_PARAMETERS):
            raise ValueError(f"covariance must have shape ({N_PARAMETERS}, {N_PARAMETERS}), got {covariance.shape}")
        if not (np.all(np.isfinite(values)) and np.all(np.isfinite(covariance))):
            raise ValueError("values and covariance must be finite")
        scale = max(float(np.abs(covariance).max()), 1e-300)
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-9 * scale):
            raise ValueError("covariance must be symmetric")
        covariance = 0.5 * (covariance + covariance.T)
        # relative to each parameter's own scale: lengths ~1e-12 m^2, angles ~1e-12 rad^2
        d = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        d[d == 0.0] = 1.0
        if np.linalg.eigvalsh(covariance / d[:, None] / d[None, :]).min() < -1e-9:
            raise ValueError("covariance must be positive semi-definite")
        values.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "covariance", covariance)

    # equality and hash by content (frozen dataclass with array fields)
    def __eq__(self, other):
        if not isinstance(other, NPLErrorParameters):
            return NotImplemented
        return self.values.tobytes() == other.values.tobytes() and self.covariance.tobytes() == other.covariance.tobytes()

    def __hash__(self):
        return hash((self.values.tobytes(), self.covariance.tobytes()))

    # -- constructors
    @classmethod
    def zero(cls) -> "NPLErrorParameters":
        """A perfect tracker: every value and the covariance zero."""
        return cls(np.zeros(N_PARAMETERS), np.zeros((N_PARAMETERS, N_PARAMETERS)), "zero")

    @classmethod
    def hughes_2011_uncompensated(cls) -> "NPLErrorParameters":
        """Table 2-3 values with a diagonal covariance from their k = 1
        uncertainties: the errors the network test found on an API T3 with its
        internal compensation switched off, in IFM mode. The correlations the
        paper mentions (gamma-b_y, a_E0-b_z, a_E2-a_E0) are not given and are
        not invented: this covariance is diagonal."""
        values, sigma = _hughes_2011_tables()
        return cls(values, np.diag(sigma**2), "Hughes 2011 Tables 2-3, API T3, uncompensated")

    @classmethod
    def hughes_2011_residual(cls) -> "NPLErrorParameters":
        """Zero mean, the same diagonal covariance: a tracker compensated *with*
        the Table 2-3 values, so what remains is their uncertainty. This is the
        one planning should use. Optimistic for a tracker between
        characterisations (drift is not in it) and a different instrument from
        any other tracker; replace with GKN's own network test."""
        _, sigma = _hughes_2011_tables()
        return cls(np.zeros(N_PARAMETERS), np.diag(sigma**2), "Hughes 2011 Tables 2-3 uncertainties, zero mean (compensated)")

    @classmethod
    def from_fit(cls, values, covariance, label: str = "fitted") -> "NPLErrorParameters":
        """From a network-test fit (`network.solve` estimate mode): the fitted
        values and their full covariance, correlations included."""
        return cls(values, covariance, label)

    # -- queries
    def is_zero(self) -> bool:
        return not (self.values.any() or self.covariance.any())

    def standard_uncertainties(self) -> np.ndarray:
        """(16,) sqrt of the covariance diagonal."""
        return np.sqrt(np.diag(self.covariance))

    def as_dict(self) -> Dict[str, float]:
        return {name: float(v) for name, v in zip(PARAMETER_NAMES, self.values)}

    # -- files
    def to_json(self, path) -> None:
        payload = {"label": self.label, "parameter_names": list(PARAMETER_NAMES),
                   "units": "metres and radians; mu dimensionless",
                   "values": self.values.tolist(), "covariance": self.covariance.tolist()}
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def from_json(cls, path) -> "NPLErrorParameters":
        """Read a parameter set written by `to_json`. The parameter names must
        match exactly, in order: a file in another order or unit is refused
        rather than guessed at."""
        payload = json.loads(Path(path).read_text())
        if tuple(payload.get("parameter_names", ())) != PARAMETER_NAMES:
            raise ValueError(f"parameter_names must be exactly {list(PARAMETER_NAMES)}")
        return cls(payload["values"], payload["covariance"], payload.get("label", Path(path).name))


def _hughes_2011_tables() -> Tuple[np.ndarray, np.ndarray]:
    """(values, standard uncertainties) in SI units, from Tables 2 and 3."""
    # (value, uncertainty), lengths in micrometres / millimetres, angles in arcseconds as printed
    table2 = {
        "lambda_m": (-2.4676e-3, 0.403e-6),
        "mu": (0.0, 5e-7),  # printed "1.000": read as 1 + mu = 1.000, i.e. mu ~ 0. TODO(source)
        "e_x_m": (4.988e-6, 0.122e-6),
        "b_y0_m": (91.201e-6, 0.654e-6),
        "b_z0_m": (-13.318e-6, 0.974e-6),
    }
    table3_arcsec = {
        "alpha_rad": (16.751, 0.128), "gamma_rad": (9.647, 0.079),
        "a_A1_rad": (0.526, 0.064), "b_A1_rad": (-0.562, 0.080),
        "a_A2_rad": (0.177, 0.073), "b_A2_rad": (1.530, 0.090),
        "a_E0_rad": (0.154, 0.223), "a_E1_rad": (-1.736, 0.152), "b_E1_rad": (-0.677, 0.183),
        "a_E2_rad": (0.375, 0.214), "b_E2_rad": (0.437, 0.179),
    }
    values, sigma = np.zeros(N_PARAMETERS), np.zeros(N_PARAMETERS)
    for name, (v, u) in table2.items():
        values[INDEX[name]], sigma[INDEX[name]] = v, u
    for name, (v, u) in table3_arcsec.items():
        values[INDEX[name]], sigma[INDEX[name]] = v * ARCSEC_TO_RAD, u * ARCSEC_TO_RAD
    return values, sigma


# -- rotations ---------------------------------------------------------------------------------
def rotation_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# -- the forward model ---------------------------------------------------------------------------
def beam_geometry(values: np.ndarray, theta_true_rad: float, phi_true_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """Where the beam leaves the instrument and where it points, at the true
    azimuth and elevation (the paper's equations 8 and 9).

    Returns `(origin_m, direction)`: the beam source point b and the unit beam
    direction n_B, in the local frame. At theta* = phi* = 0 the transit axis
    passes through e0 = (e_x, 0, 0) with direction R_x(alpha) y-hat, the source
    is at b0 = (0, b_y0, b_z0) and the beam direction is R_z(gamma) x-hat. The
    head then turns by theta* about the standing axis, and the beam is tipped
    by phi* about the (tilted, offset) transit axis:

        b   = R_z(theta*) e0 + R_z(theta*) R_x(alpha) R_y(-phi*) R_x(-alpha) (b0 - e0)
        n_B =                  R_z(theta*) R_x(alpha) R_y(-phi*) R_x(-alpha) R_z(gamma) x-hat

    With every parameter zero this is the origin and
    (cos phi cos theta, cos phi sin theta, sin phi): an ideal tracker.
    """
    e_x, b_y0, b_z0 = values[INDEX["e_x_m"]], values[INDEX["b_y0_m"]], values[INDEX["b_z0_m"]]
    alpha, gamma = values[INDEX["alpha_rad"]], values[INDEX["gamma_rad"]]
    e0 = np.array([e_x, 0.0, 0.0])
    b0 = np.array([0.0, b_y0, b_z0])
    tip = rotation_z(theta_true_rad) @ rotation_x(alpha) @ rotation_y(-phi_true_rad) @ rotation_x(-alpha)
    origin = rotation_z(theta_true_rad) @ e0 + tip @ (b0 - e0)
    direction = tip @ rotation_z(gamma) @ np.array([1.0, 0.0, 0.0])
    return origin, direction


def point_from_true(values: np.ndarray, d_true_m: float, theta_true_rad: float, phi_true_rad: float) -> np.ndarray:
    """The target position (local frame) for true range, azimuth and elevation:
    x = b + d* n_B (the paper's equation 10). Explicit -- no iteration."""
    origin, direction = beam_geometry(values, theta_true_rad, phi_true_rad)
    return origin + d_true_m * direction


def readings_from_true(values: np.ndarray, d_true_m: float, theta_true_rad: float, phi_true_rad: float) -> np.ndarray:
    """What the instrument *reports* for true range, azimuth, elevation:
    (d, theta, phi) of the paper's equations 3-5, noise-free. The azimuth is
    wrapped to (-pi, pi]; the elevation is not (a back-face reading is above
    pi/2). Fourier terms are evaluated at the true angles (see the module
    docstring)."""
    lam, mu = values[INDEX["lambda_m"]], values[INDEX["mu"]]
    a_a1, b_a1, a_a2, b_a2 = (values[INDEX[n]] for n in ("a_A1_rad", "b_A1_rad", "a_A2_rad", "b_A2_rad"))
    a_e0, a_e1, b_e1, a_e2, b_e2 = (values[INDEX[n]] for n in ("a_E0_rad", "a_E1_rad", "b_E1_rad", "a_E2_rad", "b_E2_rad"))
    d = (1.0 + mu) * d_true_m + lam
    theta = theta_true_rad + (
        a_a1 * np.cos(theta_true_rad) + b_a1 * np.sin(theta_true_rad)
        + a_a2 * np.cos(2.0 * theta_true_rad) + b_a2 * np.sin(2.0 * theta_true_rad)
    )
    phi = phi_true_rad + a_e0 + (
        a_e1 * np.cos(phi_true_rad) + b_e1 * np.sin(phi_true_rad)
        + a_e2 * np.cos(2.0 * phi_true_rad) + b_e2 * np.sin(2.0 * phi_true_rad)
    )
    return np.array([d, _wrap_to_pi(theta), phi])


def _wrap_to_pi(angle_rad: float) -> float:
    """Wrap to (-pi, pi], the range `numpy.arctan2` -- and so
    `geometry.spherical` -- uses for azimuth."""
    return float(angle_rad - 2.0 * np.pi * np.ceil((angle_rad - np.pi) / (2.0 * np.pi)))


def _wrapped_difference(a_rad: float, b_rad: float) -> float:
    return _wrap_to_pi(a_rad - b_rad)


def solve_true_coordinates(
    point_local_m, values: np.ndarray, face: str = FRONT
) -> Tuple[float, float, float, int]:
    """The true range, azimuth and elevation (d*, theta*, phi*) of a target at
    `point_local_m` for the given face, and the number of iterations used.

    Equation 10 gives the point from (d*, theta*, phi*) explicitly; going the
    other way needs a solve. Newton's method on x = b + d* n_B, starting from
    the ideal spherical solution of the same face (`geometry.spherical`) --
    which is within a few hundred micrometres and a few tens of arcseconds of
    the answer -- with a finite-difference Jacobian (a central difference of
    the explicit forward map; accuracy of the Jacobian affects the speed of
    convergence, not the answer, which is fixed by the residual), a
    minimum-norm least-squares step (so that a configuration where the ideal
    Jacobian is rank-deficient, straight up with a perfect instrument, still
    works) and step halving if a step does not reduce the residual.
    **Convergence criterion:** the beam point matches the target to
    `SOLVE_TOLERANCE_M` (0.1 pm) in every coordinate; `SOLVE_MAX_ITERATIONS`
    exceeded raises `RuntimeError`. At the Hughes-2011 magnitudes it takes
    at most 4 iterations (mean about 2) from the ideal start over elevations
    -60 to +89.9 degrees and ranges 0.27-30 m, and up to 8 for a beam pointed
    straight up (tests pin both).

    **A blind spot above the instrument.** With the beam source offset from
    the standing axis and the beam tilted, no beam passes exactly through the
    vertical axis through the instrument origin: the beams that nearly do
    all miss it by the component of the source offset perpendicular to the
    beam's tilt, of order 100 um at Hughes-2011 magnitudes. A target on that
    axis is therefore unreachable and the solve raises `RuntimeError` --
    correctly, not a failure to converge. Everything else, including the
    beam pointed straight up (target ~100 um to one side of the axis,
    which side set by the azimuth), solves normally; the paper's claim that
    the model, unlike the NIST one, has no 1/sin(elevation) terms is about
    the forward map, which stays smooth there. Azimuth itself is
    ill-conditioned near the zenith (a unit of azimuth moves the point by
    only ~d cos(phi)), inherent to any azimuth-elevation instrument.
    """
    x = np.asarray(point_local_m, dtype=float)
    if face == FRONT:
        q = np.array(cartesian_to_spherical(x))
    elif face == BACK:
        q = np.array(cartesian_to_spherical_back_face(x))
    else:
        raise ValueError(f"face must be '{FRONT}' or '{BACK}', got {face!r}")
    steps = np.array([1e-6, 1e-7, 1e-7])
    residual = point_from_true(values, *q) - x
    for iteration in range(SOLVE_MAX_ITERATIONS + 1):
        if np.max(np.abs(residual)) <= SOLVE_TOLERANCE_M:
            return float(q[0]), float(q[1]), float(q[2]), iteration
        jacobian = np.empty((3, 3))
        for k in range(3):
            shift = np.zeros(3)
            shift[k] = steps[k]
            jacobian[:, k] = (point_from_true(values, *(q + shift)) - point_from_true(values, *(q - shift))) / (2.0 * steps[k])
        step = np.linalg.lstsq(jacobian, -residual, rcond=1e-12)[0]
        scale = 1.0
        for _ in range(12):
            trial = q + scale * step
            trial_residual = point_from_true(values, *trial) - x
            if np.linalg.norm(trial_residual) < np.linalg.norm(residual):
                break
            scale *= 0.5
        q, residual = trial, trial_residual
    raise RuntimeError(
        f"true-coordinate solve did not converge in {SOLVE_MAX_ITERATIONS} iterations: the target may be "
        "on the vertical axis through the instrument origin, which no beam of an instrument with offset "
        "axes can reach (see the 'blind spot' note in solve_true_coordinates)"
    )


def true_to_observed(point_local_m, params: NPLErrorParameters, face: str = FRONT) -> np.ndarray:
    """What a tracker with these parameters *reads* for a target at
    `point_local_m` (local frame) on the given face: the noise-free
    (d, theta, phi), shape (3,). Needs the iterative solve of
    `solve_true_coordinates`. With zero parameters it equals the ideal
    spherical conversion of that face exactly."""
    d_true, theta_true, phi_true, _ = solve_true_coordinates(point_local_m, params.values, face)
    return readings_from_true(params.values, d_true, theta_true, phi_true)


def observed_to_true(d_m: float, theta_rad: float, phi_rad: float, params: NPLErrorParameters,
                     convention: str = "exact") -> np.ndarray:
    """The corrected target position (local frame) for a set of readings: the
    paper's d-hat, theta-hat, phi-hat and x-hat (section 5).

    `convention="exact"`: the exact inverse of `readings_from_true`, so
    `observed_to_true(*true_to_observed(x, p, face), p)` returns x to rounding
    error. The angle corrections are solved by fixed-point iteration (each
    pass changes the answer by (encoder error)^2: two to three passes reach
    machine precision).
    `convention="paper"`: the paper's equations 13 and 16-17 as printed, with
    the Fourier terms evaluated at the *observed* angles; differs from the
    exact inverse at second order in the encoder errors (nanometres).
    """
    v = params.values
    d_hat = (d_m - v[INDEX["lambda_m"]]) / (1.0 + v[INDEX["mu"]])
    if convention == "paper":
        theta_hat = theta_rad - _azimuth_error(v, theta_rad)
        phi_hat = phi_rad - _elevation_error(v, phi_rad)
    elif convention == "exact":
        theta_hat, phi_hat = theta_rad, phi_rad
        for _ in range(8):
            new_theta = theta_rad - _azimuth_error(v, theta_hat)
            new_phi = phi_rad - _elevation_error(v, phi_hat)
            converged = abs(new_theta - theta_hat) < 1e-16 and abs(new_phi - phi_hat) < 1e-16
            theta_hat, phi_hat = new_theta, new_phi
            if converged:
                break
    else:
        raise ValueError(f"convention must be 'exact' or 'paper', got {convention!r}")
    return point_from_true(v, d_hat, theta_hat, phi_hat)


def _azimuth_error(values: np.ndarray, theta_rad: float) -> float:
    a1, b1, a2, b2 = (values[INDEX[n]] for n in ("a_A1_rad", "b_A1_rad", "a_A2_rad", "b_A2_rad"))
    return (a1 * np.cos(theta_rad) + b1 * np.sin(theta_rad) + a2 * np.cos(2.0 * theta_rad) + b2 * np.sin(2.0 * theta_rad))


def _elevation_error(values: np.ndarray, phi_rad: float) -> float:
    a0, a1, b1, a2, b2 = (values[INDEX[n]] for n in ("a_E0_rad", "a_E1_rad", "b_E1_rad", "a_E2_rad", "b_E2_rad"))
    return a0 + a1 * np.cos(phi_rad) + b1 * np.sin(phi_rad) + a2 * np.cos(2.0 * phi_rad) + b2 * np.sin(2.0 * phi_rad)


# -- sensitivities -----------------------------------------------------------------------------
def _reading_difference(plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    out = plus - minus
    out[1] = _wrapped_difference(plus[1], minus[1])
    return out


def observation_point_jacobian(point_local_m, params: NPLErrorParameters, face: str = FRONT, step_m: float = 1e-6) -> np.ndarray:
    """(3, 3) d(d, theta, phi) / d(x, y, z): how the readings change when the
    target moves, by central finite differences of `true_to_observed`
    (step 1 um: far above the 1e-13 m solve tolerance, far below any scale
    on which the mapping curves). For a perfect tracker, front face, it is
    the inverse of `geometry.spherical.spherical_jacobian`."""
    x = np.asarray(point_local_m, dtype=float)
    jacobian = np.empty((3, 3))
    for k in range(3):
        shift = np.zeros(3)
        shift[k] = step_m
        jacobian[:, k] = _reading_difference(true_to_observed(x + shift, params, face),
                                             true_to_observed(x - shift, params, face)) / (2.0 * step_m)
    return jacobian


def observation_parameter_jacobian(point_local_m, params: NPLErrorParameters, face: str = FRONT,
                                   steps: np.ndarray = None) -> np.ndarray:
    """(3, 16) d(d, theta, phi) / d(parameters) for a fixed target position:
    how much each reading moves per unit error in each parameter, at the
    parameter values of `params`, by central differences with the per-parameter
    steps of `PARAMETER_STEPS` (justified there). Columns in `PARAMETER_NAMES`
    order, SI units (metres, radians). This is the matrix the network layers
    multiply the parameter covariance by. `steps` (16,) overrides
    `PARAMETER_STEPS`, to check a result does not depend on the step."""
    x = np.asarray(point_local_m, dtype=float)
    steps = PARAMETER_STEPS if steps is None else np.asarray(steps, dtype=float)
    jacobian = np.empty((3, N_PARAMETERS))
    for k in range(N_PARAMETERS):
        step = steps[k]
        plus, minus = params.values.copy(), params.values.copy()
        plus[k] += step
        minus[k] -= step
        jacobian[:, k] = _reading_difference(
            true_to_observed(x, NPLErrorParameters(plus, np.zeros((N_PARAMETERS, N_PARAMETERS))), face),
            true_to_observed(x, NPLErrorParameters(minus, np.zeros((N_PARAMETERS, N_PARAMETERS))), face),
        ) / (2.0 * step)
    return jacobian
