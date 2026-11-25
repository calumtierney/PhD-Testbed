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


# =============================================================================================
# Batch versions, for the network layers: hundreds of shots per layout. The scalar functions above
# stay the plain, readable statement of the physics; each batch function is tested against them.
# `values` is one parameter vector (16,) shared by every shot, or one per shot (K, 16) when the
# shots come from different physical trackers.
# =============================================================================================
def _batch_values(values, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.broadcast_to(values, (k, N_PARAMETERS)) if values.ndim == 1 else values


def _rotation_batch(axis: str, angle_rad: np.ndarray) -> np.ndarray:
    """(K,) angles -> (K, 3, 3) rotations about x, y or z (the same matrices as `rotation_x/y/z`)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    out = np.zeros(angle_rad.shape + (3, 3))
    if axis == "x":
        out[..., 0, 0] = 1.0
        out[..., 1, 1], out[..., 1, 2], out[..., 2, 1], out[..., 2, 2] = c, -s, s, c
    elif axis == "y":
        out[..., 1, 1] = 1.0
        out[..., 0, 0], out[..., 0, 2], out[..., 2, 0], out[..., 2, 2] = c, s, -s, c
    else:
        out[..., 2, 2] = 1.0
        out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1] = c, -s, s, c
    return out


def beam_geometry_batch(values, theta_true_rad: np.ndarray, phi_true_rad: np.ndarray):
    """`beam_geometry` for K shots: `(origin (K,3), direction (K,3))`."""
    theta = np.asarray(theta_true_rad, dtype=float)
    k = theta.shape[0]
    v = _batch_values(values, k)
    alpha, gamma = v[:, INDEX["alpha_rad"]], v[:, INDEX["gamma_rad"]]
    e0 = np.zeros((k, 3))
    e0[:, 0] = v[:, INDEX["e_x_m"]]
    b0 = np.zeros((k, 3))
    b0[:, 1], b0[:, 2] = v[:, INDEX["b_y0_m"]], v[:, INDEX["b_z0_m"]]
    rz_theta = _rotation_batch("z", theta)
    tip = rz_theta @ _rotation_batch("x", alpha) @ _rotation_batch("y", -np.asarray(phi_true_rad, dtype=float)) @ _rotation_batch("x", -alpha)
    origin = np.einsum("kij,kj->ki", rz_theta, e0) + np.einsum("kij,kj->ki", tip, b0 - e0)
    x_hat = np.zeros((k, 3))
    x_hat[:, 0] = 1.0
    direction = np.einsum("kij,kj->ki", tip @ _rotation_batch("z", gamma), x_hat)
    return origin, direction


def point_from_true_batch(values, d_true_m, theta_true_rad, phi_true_rad) -> np.ndarray:
    """`point_from_true` for K shots: (K, 3)."""
    origin, direction = beam_geometry_batch(values, theta_true_rad, phi_true_rad)
    return origin + np.asarray(d_true_m, dtype=float)[:, None] * direction


def readings_from_true_batch(values, d_true_m, theta_true_rad, phi_true_rad) -> np.ndarray:
    """`readings_from_true` for K shots: (K, 3) of (d, theta, phi), azimuth wrapped to (-pi, pi]."""
    d_true, theta, phi = (np.asarray(a, dtype=float) for a in (d_true_m, theta_true_rad, phi_true_rad))
    v = _batch_values(values, d_true.shape[0])
    g = lambda name: v[:, INDEX[name]]
    d = (1.0 + g("mu")) * d_true + g("lambda_m")
    theta_read = theta + (g("a_A1_rad") * np.cos(theta) + g("b_A1_rad") * np.sin(theta)
                          + g("a_A2_rad") * np.cos(2 * theta) + g("b_A2_rad") * np.sin(2 * theta))
    phi_read = phi + g("a_E0_rad") + (g("a_E1_rad") * np.cos(phi) + g("b_E1_rad") * np.sin(phi)
                                      + g("a_E2_rad") * np.cos(2 * phi) + g("b_E2_rad") * np.sin(2 * phi))
    theta_read = theta_read - 2.0 * np.pi * np.ceil((theta_read - np.pi) / (2.0 * np.pi))
    return np.stack([d, theta_read, phi_read], axis=1)


def ideal_readings_batch(points_local_m: np.ndarray, back: np.ndarray) -> np.ndarray:
    """The ideal (d, theta, phi) of K points, front face where `back` is False and
    back face where True (`geometry.spherical`'s conversions, vectorised)."""
    p = np.asarray(points_local_m, dtype=float)
    d = np.linalg.norm(p, axis=1)
    theta = np.arctan2(p[:, 1], p[:, 0])
    phi = np.arcsin(np.clip(p[:, 2] / d, -1.0, 1.0))
    back = np.asarray(back, dtype=bool)
    theta = np.where(back, theta + np.pi, theta)
    theta = np.where(theta > np.pi, theta - 2.0 * np.pi, theta)
    phi = np.where(back, np.pi - phi, phi)
    return np.stack([d, theta, phi], axis=1)


def _faces_to_back(faces, k: int) -> np.ndarray:
    if faces is None:
        return np.zeros(k, dtype=bool)
    arr = np.asarray(faces)
    if arr.ndim == 0:
        arr = np.full(k, arr)
    bad = [f for f in np.unique(arr) if f not in (FRONT, BACK)]
    if bad:
        raise ValueError(f"face must be '{FRONT}' or '{BACK}', got {bad}")
    return arr == BACK


def solve_true_coordinates_batch(points_local_m: np.ndarray, values, faces=None) -> np.ndarray:
    """`solve_true_coordinates` for K shots at once: (K, 3) of (d*, theta*, phi*).

    The same Newton iteration, from the same ideal start, on every shot
    together (a pseudo-inverse step, so the rank-deficient straight-up case of a
    perfect instrument still works); the same 0.1 pm convergence criterion.
    Shots that have not converged -- a target in the blind spot, or an
    extremely ill-conditioned one -- are handed to the scalar solver, which
    either solves them with step halving or raises its explanatory error.
    """
    points = np.asarray(points_local_m, dtype=float)
    k = points.shape[0]
    v = _batch_values(values, k)
    back = _faces_to_back(faces, k)
    q = ideal_readings_batch(points, back)
    steps = np.array([1e-6, 1e-7, 1e-7])
    for _ in range(SOLVE_MAX_ITERATIONS):
        residual = point_from_true_batch(v, *q.T) - points
        active = np.max(np.abs(residual), axis=1) > SOLVE_TOLERANCE_M
        if not active.any():
            return q
        jac = np.empty((k, 3, 3))
        for col in range(3):
            shift = np.zeros(3)
            shift[col] = steps[col]
            jac[:, :, col] = (point_from_true_batch(v, *(q + shift).T) - point_from_true_batch(v, *(q - shift).T)) / (2.0 * steps[col])
        step = -np.einsum("kij,kj->ki", np.linalg.pinv(jac, rcond=1e-12), residual)
        q = q + np.where(active[:, None], step, 0.0)
    residual = point_from_true_batch(v, *q.T) - points
    for i in np.nonzero(np.max(np.abs(residual), axis=1) > SOLVE_TOLERANCE_M)[0]:
        d, th, ph, _ = solve_true_coordinates(points[i], v[i], BACK if back[i] else FRONT)  # raises, with its explanation, if unreachable
        q[i] = (d, th, ph)
    return q


def true_to_observed_batch(points_local_m: np.ndarray, values, faces=None) -> np.ndarray:
    """`true_to_observed` for K shots: (K, 3) readings."""
    q = solve_true_coordinates_batch(points_local_m, values, faces)
    return readings_from_true_batch(values, *q.T)


def reading_sensitivities(points_local_m: np.ndarray, values, faces=None):
    """Sensitivities of K shots, at the parameter values given, from the explicit
    forward maps only (no finite differences through the iterative solve):

      reading_wrt_point  (K, 3, 3)   d(d, theta, phi) / d(x, y, z)
      reading_wrt_params (K, 3, 16)  d(d, theta, phi) / d(parameters), target fixed
      point_wrt_params   (K, 3, 16)  P: the shift of the point an *ideal* conversion
          of the corrected readings would report, per unit error in each
          parameter, target fixed (local frame, metres per unit)

    Write x = F(q; c) for the target position at true coordinates q =
    (d*, theta*, phi*), and r = G(q; c) for the readings. Holding the target
    fixed while a parameter changes moves q by -F_q^-1 F_c, so

        dr/dx = G_q F_q^-1,        dr/dc = G_c - G_q F_q^-1 F_c,
        P = (dr/dx)^-1 dr/dc = F_q G_q^-1 G_c - F_c,

    all four derivatives being of explicit functions, taken by central
    differences with the steps of `PARAMETER_STEPS` for c and 1 um / 0.1 urad
    for q. At zero parameters F_q is `geometry.spherical.spherical_jacobian`,
    G_q the identity, and P reduces to J_sph G_c - F_c: a parameter that
    reads the angle wrongly (encoder) moves the reported point through the
    spherical Jacobian, one that moves the beam (axis offsets and tilts) moves
    it the other way. P is what the network layers need: a parameter error is,
    to the network, a shift of every shot's pseudo-measurement by P times it.
    """
    points = np.asarray(points_local_m, dtype=float)
    k = points.shape[0]
    v = _batch_values(values, k)
    q = solve_true_coordinates_batch(points, v, faces)
    q_steps = np.array([1e-6, 1e-6, 1e-6])
    f_q, g_q = np.empty((k, 3, 3)), np.empty((k, 3, 3))
    for col in range(3):
        shift = np.zeros(3)
        shift[col] = q_steps[col]
        plus, minus = q + shift, q - shift
        f_q[:, :, col] = (point_from_true_batch(v, *plus.T) - point_from_true_batch(v, *minus.T)) / (2.0 * q_steps[col])
        diff = readings_from_true_batch(v, *plus.T) - readings_from_true_batch(v, *minus.T)
        diff[:, 1] = (diff[:, 1] + np.pi) % (2 * np.pi) - np.pi
        g_q[:, :, col] = diff / (2.0 * q_steps[col])
    f_c, g_c = np.empty((k, 3, N_PARAMETERS)), np.empty((k, 3, N_PARAMETERS))
    for col in range(N_PARAMETERS):
        plus, minus = v.copy(), v.copy()
        plus[:, col] += PARAMETER_STEPS[col]
        minus[:, col] -= PARAMETER_STEPS[col]
        f_c[:, :, col] = (point_from_true_batch(plus, *q.T) - point_from_true_batch(minus, *q.T)) / (2.0 * PARAMETER_STEPS[col])
        diff = readings_from_true_batch(plus, *q.T) - readings_from_true_batch(minus, *q.T)
        diff[:, 1] = (diff[:, 1] + np.pi) % (2 * np.pi) - np.pi
        g_c[:, :, col] = diff / (2.0 * PARAMETER_STEPS[col])
    g_q_inv_g_c = np.linalg.solve(g_q, g_c)
    f_q_inv_f_c = np.linalg.solve(f_q, f_c)
    reading_wrt_point = g_q @ np.linalg.inv(f_q)
    reading_wrt_params = g_c - g_q @ f_q_inv_f_c
    point_wrt_params = f_q @ g_q_inv_g_c - f_c
    return reading_wrt_point, reading_wrt_params, point_wrt_params


def shot_shift_zero_mean(points_local_m: np.ndarray, faces=None) -> np.ndarray:
    """`reading_sensitivities(...)[2]` (the point shift P per unit parameter
    error) for parameters whose *values* are zero -- the compensated tracker of
    `NPLErrorParameters.hughes_2011_residual` -- in closed form, vectorised, no
    finite differences. About a fiftieth of the cost, which is what a station
    search needs.

    At zero parameters F_q is the spherical Jacobian J of the face's ideal
    angles (d, theta, phi) and G_q the identity, so P = J G_c - F_c, with

      G_c (the readings' direct sensitivity): lambda -> (1,0,0), mu -> (d,0,0),
        azimuth harmonics a_A,q b_A,q -> (0, cos q theta | sin q theta, 0),
        elevation terms a_E,0 -> (0,0,1) and a_E,q b_E,q -> (0,0, cos q phi |
        sin q phi); zero for the five geometric parameters;
      F_c (the beam's own displacement, T0 = R_z(theta) R_y(-phi)):
        e_x  -> R_z(theta) (1 - cos phi, 0, -sin phi)
        b_y0 -> T0 y-hat = (-sin theta, cos theta, 0)
        b_z0 -> T0 z-hat = (-sin phi cos theta, -sin phi sin theta, cos phi)
        gamma -> d T0 y-hat
        alpha -> d sin phi (sin theta, -cos theta, 0);  zero for the rest.

    The same numbers, to 1e-6, as the finite-difference route (tests).
    """
    points = np.asarray(points_local_m, dtype=float)
    k = points.shape[0]
    d, theta, phi = ideal_readings_batch(points, _faces_to_back(faces, k)).T
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    jac = np.zeros((k, 3, 3))  # the spherical Jacobian at the face's ideal angles
    jac[:, :, 0] = np.stack([cp * ct, cp * st, sp], axis=1)
    jac[:, :, 1] = d[:, None] * np.stack([-cp * st, cp * ct, np.zeros(k)], axis=1)
    jac[:, :, 2] = d[:, None] * np.stack([-sp * ct, -sp * st, cp], axis=1)
    g_c = np.zeros((k, 3, N_PARAMETERS))
    g_c[:, 0, INDEX["lambda_m"]] = 1.0
    g_c[:, 0, INDEX["mu"]] = d
    g_c[:, 1, INDEX["a_A1_rad"]], g_c[:, 1, INDEX["b_A1_rad"]] = ct, st
    g_c[:, 1, INDEX["a_A2_rad"]], g_c[:, 1, INDEX["b_A2_rad"]] = np.cos(2 * theta), np.sin(2 * theta)
    g_c[:, 2, INDEX["a_E0_rad"]] = 1.0
    g_c[:, 2, INDEX["a_E1_rad"]], g_c[:, 2, INDEX["b_E1_rad"]] = cp, sp
    g_c[:, 2, INDEX["a_E2_rad"]], g_c[:, 2, INDEX["b_E2_rad"]] = np.cos(2 * phi), np.sin(2 * phi)
    f_c = np.zeros((k, 3, N_PARAMETERS))
    f_c[:, :, INDEX["e_x_m"]] = np.stack([ct * (1 - cp), st * (1 - cp), -sp], axis=1)
    f_c[:, :, INDEX["b_y0_m"]] = np.stack([-st, ct, np.zeros(k)], axis=1)
    f_c[:, :, INDEX["b_z0_m"]] = np.stack([-sp * ct, -sp * st, cp], axis=1)
    f_c[:, :, INDEX["gamma_rad"]] = d[:, None] * np.stack([-st, ct, np.zeros(k)], axis=1)
    f_c[:, :, INDEX["alpha_rad"]] = d[:, None] * sp[:, None] * np.stack([st, -ct, np.zeros(k)], axis=1)
    return jac @ g_c - f_c
