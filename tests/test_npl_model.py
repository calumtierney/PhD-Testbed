"""The NPL geometric error model (`instruments.npl_model`): gates 1-4 of the
full-model task. Where a paper's number exists it is checked; the two-face
laws are derived independently of the paper's numbers (docstring of
`test_two_face_laws`) so a sign or frame error fails here, not in a figure.
"""
import pickle

import numpy as np
import pytest

from geometry.spherical import (
    cartesian_to_spherical, cartesian_to_spherical_back_face, spherical_jacobian, spherical_to_cartesian,
)
from instruments.laser_tracker import LaserTracker
from instruments.npl_model import (
    ARCSEC_TO_RAD, BACK, FRONT, INDEX, N_PARAMETERS, PARAMETER_NAMES, PARAMETER_STEPS, NPLErrorParameters,
    beam_geometry, beam_geometry_batch, observation_parameter_jacobian, observation_point_jacobian, observed_to_true,
    point_from_true, point_from_true_batch, readings_from_true, readings_from_true_batch, reading_sensitivities,
    solve_true_coordinates, solve_true_coordinates_batch, true_to_observed, true_to_observed_batch,
)

AS = ARCSEC_TO_RAD
HUGHES = NPLErrorParameters.hughes_2011_uncompensated()
ZERO = NPLErrorParameters.zero()
ELEVATIONS_DEG = (-60.0, -30.0, 0.0, 30.0, 57.0, 77.0)  # the tracker's mechanical range is -60 to +77 deg


def only(name: str, value: float) -> NPLErrorParameters:
    values = np.zeros(N_PARAMETERS)
    values[INDEX[name]] = value
    return NPLErrorParameters(values, np.zeros((N_PARAMETERS, N_PARAMETERS)))


# -- the parameter set -----------------------------------------------------------------------------
def test_hughes_tables_are_transcribed_correctly():
    """Every number of Tables 2 and 3 (value, k=1 uncertainty), in the units
    the paper prints them in, against the checked PDF."""
    printed = {  # name: (value, uncertainty, unit) -- unit 'm' or 'arcsec' (mu dimensionless)
        "lambda_m": (-2.4676e-3, 0.403e-6, 1.0), "mu": (0.0, 5e-7, 1.0),
        "e_x_m": (4.988e-6, 0.122e-6, 1.0), "b_y0_m": (91.201e-6, 0.654e-6, 1.0), "b_z0_m": (-13.318e-6, 0.974e-6, 1.0),
        "alpha_rad": (16.751, 0.128, AS), "gamma_rad": (9.647, 0.079, AS),
        "a_A1_rad": (0.526, 0.064, AS), "b_A1_rad": (-0.562, 0.080, AS), "a_A2_rad": (0.177, 0.073, AS),
        "b_A2_rad": (1.530, 0.090, AS), "a_E0_rad": (0.154, 0.223, AS), "a_E1_rad": (-1.736, 0.152, AS),
        "b_E1_rad": (-0.677, 0.183, AS), "a_E2_rad": (0.375, 0.214, AS), "b_E2_rad": (0.437, 0.179, AS),
    }
    assert set(printed) == set(PARAMETER_NAMES) and len(PARAMETER_NAMES) == N_PARAMETERS == 16
    sigma = HUGHES.standard_uncertainties()
    for name, (value, uncertainty, unit) in printed.items():
        assert HUGHES.values[INDEX[name]] == pytest.approx(value * unit, rel=1e-12, abs=0.0), name
        assert sigma[INDEX[name]] == pytest.approx(uncertainty * unit, rel=1e-12), name
    assert np.count_nonzero(HUGHES.covariance - np.diag(np.diag(HUGHES.covariance))) == 0  # no invented correlations


def test_residual_is_zero_mean_with_the_same_covariance_and_zero_is_zero():
    residual = NPLErrorParameters.hughes_2011_residual()
    assert not residual.values.any()
    np.testing.assert_array_equal(residual.covariance, HUGHES.covariance)
    assert ZERO.is_zero() and not HUGHES.is_zero() and not residual.is_zero()


def test_parameter_sets_are_immutable_comparable_picklable_and_json_round_trip(tmp_path):
    with pytest.raises(ValueError):
        HUGHES.values[0] = 1.0
    assert HUGHES == NPLErrorParameters.hughes_2011_uncompensated() and hash(HUGHES) == hash(NPLErrorParameters.hughes_2011_uncompensated())
    assert HUGHES != ZERO
    assert pickle.loads(pickle.dumps(HUGHES)) == HUGHES
    HUGHES.to_json(tmp_path / "tracker.json")
    assert NPLErrorParameters.from_json(tmp_path / "tracker.json") == HUGHES
    fitted = NPLErrorParameters.from_fit(HUGHES.values, HUGHES.covariance, "my network test")
    assert fitted.label == "my network test" and fitted == HUGHES


def test_parameter_set_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        NPLErrorParameters(np.zeros(15), np.zeros((16, 16)))
    asymmetric = np.zeros((16, 16))
    asymmetric[0, 1] = 1e-12
    with pytest.raises(ValueError, match="symmetric"):
        NPLErrorParameters(np.zeros(16), asymmetric)
    indefinite = np.eye(16) * 1e-12
    indefinite[0, 1] = indefinite[1, 0] = 2e-12
    with pytest.raises(ValueError, match="positive"):
        NPLErrorParameters(np.zeros(16), indefinite)
    with pytest.raises(ValueError, match="finite"):
        NPLErrorParameters(np.full(16, np.nan), np.zeros((16, 16)))
    HUGHES.to_json(tmp_path / "t.json")
    text = (tmp_path / "t.json").read_text().replace("lambda_m", "lambda_mm")
    (tmp_path / "wrong.json").write_text(text)
    with pytest.raises(ValueError, match="parameter_names"):
        NPLErrorParameters.from_json(tmp_path / "wrong.json")


# -- gate 1: nothing changes when the model is off -----------------------------------------------------
def test_tracker_without_geometric_params_is_unchanged_and_nonzero_ones_are_not_silently_ignored():
    point = np.array([2.5, 0.3, 0.4])
    base = LaserTracker().covariance_local(point)
    np.testing.assert_array_equal(LaserTracker(geometric_params=ZERO).covariance_local(point), base)
    assert LaserTracker().geometric_params is None
    with pytest.raises(NotImplementedError, match="network layer"):
        LaserTracker(geometric_params=HUGHES).covariance_local(point)
    assert LaserTracker() == LaserTracker() and LaserTracker(geometric_params=HUGHES) == LaserTracker(geometric_params=HUGHES)
    assert LaserTracker(geometric_params=HUGHES) != LaserTracker()
    assert pickle.loads(pickle.dumps(LaserTracker(geometric_params=HUGHES))) == LaserTracker(geometric_params=HUGHES)


# -- gate 2: with every parameter zero the model is the spherical conversion ---------------------------------
def test_zero_parameters_give_the_ideal_beam():
    for theta in np.linspace(-np.pi, np.pi, 13):
        for phi in np.deg2rad(np.linspace(-80, 80, 9)):
            origin, direction = beam_geometry(ZERO.values, theta, phi)
            np.testing.assert_allclose(origin, 0.0, atol=1e-16)
            np.testing.assert_allclose(direction, [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)], atol=1e-15)


@pytest.mark.parametrize("face", [FRONT, BACK])
def test_zero_parameters_reproduce_the_spherical_conversion_on_both_faces(face):
    worst = 0.0
    for d in (0.27, 2.5, 6.4):
        for az in np.linspace(-np.pi, np.pi, 13)[:-1]:
            for el in np.deg2rad(ELEVATIONS_DEG):
                x = spherical_to_cartesian(d, az, el)
                reading = true_to_observed(x, ZERO, face)
                ideal = cartesian_to_spherical(x) if face == FRONT else cartesian_to_spherical_back_face(x)
                azimuth_error = abs((reading[1] - ideal[1] + np.pi) % (2 * np.pi) - np.pi)  # on the circle
                worst = max(worst, abs(reading[0] - ideal[0]), azimuth_error, abs(reading[2] - ideal[2]))
                np.testing.assert_allclose(observed_to_true(*reading, ZERO), x, atol=1e-14)
    assert worst < 1e-13


def test_back_face_reading_is_the_same_point_reached_over_the_top():
    x = np.array([1.2, -0.7, 0.9])
    d, theta, phi = cartesian_to_spherical(x)
    d_b, theta_b, phi_b = cartesian_to_spherical_back_face(x)
    assert d_b == d and np.pi / 2 < phi_b < 3 * np.pi / 2 and phi_b == pytest.approx(np.pi - phi)
    assert abs(((theta_b - theta) % (2 * np.pi)) - np.pi) < 1e-12 and -np.pi < theta_b <= np.pi
    np.testing.assert_allclose(spherical_to_cartesian(d_b, theta_b, phi_b), x, atol=1e-14)
    with pytest.raises(ValueError):
        cartesian_to_spherical_back_face([0.0, 0.0, 0.0])


# -- gate 3: round trip, including near-vertical beams -----------------------------------------------------
@pytest.mark.parametrize("face", [FRONT, BACK])
def test_round_trip_is_exact_with_the_uncompensated_hughes_values(face):
    """true -> observed -> true returns the point to far better than the 1 nm
    gate (picometres: rounding) at ranges 0.27-30 m, elevations -60 to 89.9
    degrees, every azimuth."""
    worst, iterations = 0.0, []
    for d in (0.27, 2.5, 6.4, 30.0):
        for az in np.linspace(-np.pi, np.pi, 13)[:-1]:
            for el in np.deg2rad(ELEVATIONS_DEG + (89.9,)):
                x = spherical_to_cartesian(d, az, el)
                iterations.append(solve_true_coordinates(x, HUGHES.values, face)[3])
                worst = max(worst, np.abs(observed_to_true(*true_to_observed(x, HUGHES, face), HUGHES) - x).max())
    assert worst < 1e-12 and max(iterations) <= 5


@pytest.mark.parametrize("face, theta_shift, phi", [(FRONT, 0.0, np.pi / 2 - 1e-5), (FRONT, 0.0, np.pi / 2),
                                                    (BACK, np.pi, np.pi / 2), (BACK, np.pi, np.pi / 2 + 1e-5)])
def test_round_trip_with_the_beam_pointing_straight_up(face, theta_shift, phi):
    """A beam at true elevation 90 degrees reaches a point ~100 um to one side
    of the vertical axis. It solves and round-trips normally -- the model has
    no singularity there, as the paper says."""
    for d in (0.27, 3.0, 6.4):
        for theta in np.linspace(-3.0, 3.0, 13):
            x = point_from_true(HUGHES.values, d, theta + theta_shift, phi)
            _, _, _, iterations = solve_true_coordinates(x, HUGHES.values, face)
            assert iterations <= 10
            assert np.abs(observed_to_true(*true_to_observed(x, HUGHES, face), HUGHES) - x).max() < 1e-12


def test_a_target_exactly_on_the_vertical_axis_is_unreachable_and_says_so():
    """The 'blind spot': no beam of an instrument with offset axes passes
    through the vertical axis through its origin. A perfect instrument can."""
    on_axis = np.array([0.0, 0.0, 3.0])
    with pytest.raises(RuntimeError, match="vertical axis"):
        solve_true_coordinates(on_axis, HUGHES.values, FRONT)
    d, theta, phi, _ = solve_true_coordinates(on_axis, ZERO.values, FRONT)
    assert (d, phi) == (3.0, pytest.approx(np.pi / 2))


def test_round_trip_survives_ten_times_the_hughes_errors():
    rng = np.random.default_rng(0)
    sigma = HUGHES.standard_uncertainties()
    for _ in range(60):
        params = NPLErrorParameters(10.0 * (HUGHES.values + rng.normal(0.0, 1.0, N_PARAMETERS) * sigma), np.zeros((16, 16)))
        x = spherical_to_cartesian(rng.uniform(0.3, 6.4), rng.uniform(-np.pi, np.pi), np.deg2rad(rng.uniform(-60, 77)))
        for face in (FRONT, BACK):
            assert np.abs(observed_to_true(*true_to_observed(x, params, face), params) - x).max() < 1e-12


def test_paper_correction_convention_differs_from_the_exact_inverse_by_nanometres():
    """Equations 13 and 16-17 evaluate the Fourier terms at the observed
    angle, the forward model at the true one; the difference is second order
    in the encoder errors: under 1 nm out to the paper's 6.4 m and a few nm
    at 30 m, a thousandth of the 2-5 um uncertainty of the points."""
    worst = {6.4: 0.0, 30.0: 0.0}
    for d in worst:
        for face in (FRONT, BACK):
            for az in np.linspace(-np.pi, np.pi, 25)[:-1]:
                for el in np.deg2rad(np.linspace(-60, 77, 12)):
                    x = spherical_to_cartesian(d, az, el)
                    r = true_to_observed(x, HUGHES, face)
                    worst[d] = max(worst[d], np.abs(observed_to_true(*r, HUGHES, convention="paper") - x).max())
    assert worst[6.4] < 1e-9 and worst[30.0] < 5e-9
    assert worst[6.4] > 1e-11  # it is a real, if tiny, difference
    with pytest.raises(ValueError):
        observed_to_true(1.0, 0.0, 0.0, HUGHES, convention="other")


def test_existing_spherical_jacobian_is_singular_at_the_zenith_but_the_npl_forward_map_is_not():
    """Reported, not fixed: `geometry.spherical.spherical_jacobian` loses a
    column at 90 degrees elevation (the azimuth column is proportional to
    cos phi), so the ideal point->reading map is singular there. The NPL
    forward map's azimuth column is small (a few hundred micrometres per
    radian at 3 m, versus 3 m per radian away from the zenith) but not zero:
    the offsets and tilts keep azimuth defined, if barely."""
    ideal = spherical_jacobian(3.0, 0.4, np.pi / 2)
    assert abs(np.linalg.det(ideal)) < 1e-12 and np.linalg.norm(ideal[:, 1]) < 1e-15
    q = np.array([3.0, 0.4, np.pi / 2])
    column = (point_from_true(HUGHES.values, *(q + [0, 1e-7, 0])) - point_from_true(HUGHES.values, *(q - [0, 1e-7, 0]))) / 2e-7
    assert 1e-5 < np.linalg.norm(column) < 1e-3


# -- gate 4: two-face physics, independent of the paper's numbers -------------------------------------------
def two_face(point, params):
    """(azimuth two-face difference, elevation two-face sum) in radians:
    (theta_back - pi) - theta_front, and phi_front + phi_back - pi. Zero for
    a perfect instrument."""
    front = true_to_observed(point, params, FRONT)
    back = true_to_observed(point, params, BACK)
    difference = (back[1] - np.pi) - front[1]
    return (difference + np.pi) % (2 * np.pi) - np.pi, front[2] + back[2] - np.pi


def grid(ranges=(1.0, 3.0, 6.0), azimuths=(-2.0, 0.7, 2.5)):
    for d in ranges:
        for az in azimuths:
            for el_deg in ELEVATIONS_DEG:
                yield d, az, np.deg2rad(el_deg), spherical_to_cartesian(d, az, np.deg2rad(el_deg))


def test_perfect_tracker_has_no_two_face_difference():
    for _, _, _, x in grid():
        assert two_face(x, ZERO) == pytest.approx((0.0, 0.0), abs=1e-14)


def test_two_face_laws():
    """Each geometric parameter alone, at +-10 arcsec or 100 um, produces a
    two-face signature with a closed form (first order in the parameter; the
    tolerance is the second-order remainder). Derivations, with the head's
    elevation tilt applied to the beam direction v:

    collimation gamma. Before tipping, the beam is x-hat + gamma y-hat. Tipping
      by phi about the (horizontal) transit axis gives (cos phi, gamma,
      sin phi): azimuth +gamma/cos phi off the commanded one, so the front-face
      reading is theta_true - gamma/cos phi. The back face tips by pi - phi
      and turns the head by pi: horizontal part (-cos phi, gamma), azimuth
      +pi - gamma/cos phi, so its reading (minus pi) is theta_true +
      gamma/cos phi. Difference **+2 gamma / cos phi**: the theodolite
      collimation-error law.
    transit-axis tilt alpha. Tipping x-hat about the axis (0, 1, alpha) by
      -phi (Rodrigues) gives (cos phi, -alpha sin phi, sin phi): azimuth
      -alpha tan phi off, front reading theta_true + alpha tan phi; the back
      face gives theta_true - alpha tan phi. Difference **-2 alpha tan phi**:
      the trunnion-axis law, opposite in sign to collimation at positive
      elevation in this model's convention.
    beam offset b_y0: lateral offset of the beam by b_y0 at range d subtends
      b_y0 / (d cos phi) in azimuth, with opposite sign on the other face:
      difference 2 b_y0 / (d cos phi). It falls with range, unlike gamma.
    Elevation sum: a_E0 appears equally on both faces (sum 2 a_E0); the
      vertical beam offset b_z0 flips sign between faces (sum -2 b_z0 / d);
      the transit-axis offset e_x gives 2 e_x sin phi / d; the encoder
      harmonics give 2 b_E1 sin phi and 2 a_E2 cos 2 phi (the cos phi and
      sin 2 phi harmonics cancel between faces), and the azimuth harmonics
      -2 (a_A1 cos theta + b_A1 sin theta) (the second harmonics cancel).
    The parameters that do not appear in a two-face test at all (lambda, mu,
    a_A2, b_A2, a_E1, b_E2) are checked to give zero.
    """
    laws = {
        # name: (value, azimuth difference law, elevation sum law)
        "gamma_rad": (10 * AS, lambda d, az, el: 2 * 10 * AS / np.cos(el), lambda d, az, el: 0.0),
        "alpha_rad": (10 * AS, lambda d, az, el: -2 * 10 * AS * np.tan(el), lambda d, az, el: 0.0),
        "b_y0_m": (100e-6, lambda d, az, el: 2 * 100e-6 / (d * np.cos(el)), lambda d, az, el: 0.0),
        "e_x_m": (100e-6, lambda d, az, el: 0.0, lambda d, az, el: 2 * 100e-6 * np.sin(el) / d),
        "b_z0_m": (100e-6, lambda d, az, el: 0.0, lambda d, az, el: -2 * 100e-6 / d),
        "a_E0_rad": (10 * AS, lambda d, az, el: 0.0, lambda d, az, el: 2 * 10 * AS),
        "b_E1_rad": (10 * AS, lambda d, az, el: 0.0, lambda d, az, el: 2 * 10 * AS * np.sin(el)),
        "a_E2_rad": (10 * AS, lambda d, az, el: 0.0, lambda d, az, el: 2 * 10 * AS * np.cos(2 * el)),
        "a_A1_rad": (10 * AS, lambda d, az, el: -2 * 10 * AS * np.cos(az), lambda d, az, el: 0.0),
        "b_A1_rad": (10 * AS, lambda d, az, el: -2 * 10 * AS * np.sin(az), lambda d, az, el: 0.0),
    }
    for name, (value, azimuth_law, elevation_law) in laws.items():
        params = only(name, value)
        for d, az, el, x in grid():
            difference, total = two_face(x, params)
            assert difference == pytest.approx(azimuth_law(d, az, el), abs=2e-3 * AS), (name, d, az, np.rad2deg(el))
            assert total == pytest.approx(elevation_law(d, az, el), abs=2e-3 * AS), (name, d, az, np.rad2deg(el))
    for name in ("lambda_m", "mu", "a_A2_rad", "b_A2_rad", "a_E1_rad", "b_E2_rad"):
        params = only(name, 1e-4 if name in ("lambda_m", "mu") else 10 * AS)
        for _, _, _, x in grid():
            assert two_face(x, params) == pytest.approx((0.0, 0.0), abs=1e-9 * AS)


def test_single_face_signs_of_collimation_and_trunnion_errors():
    """The signs behind the two-face laws, face by face: the azimuth reading
    minus the true azimuth (front), and the back reading minus pi minus the
    true azimuth, for 10 arcsec of each."""
    for _, az, el, x in grid():
        true_az = cartesian_to_spherical(x)[1]
        for name, front_law, back_law in (
            ("gamma_rad", -10 * AS / np.cos(el), +10 * AS / np.cos(el)),
            ("alpha_rad", +10 * AS * np.tan(el), -10 * AS * np.tan(el)),
        ):
            params = only(name, 10 * AS)
            front, back = true_to_observed(x, params, FRONT), true_to_observed(x, params, BACK)
            wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
            assert wrap(front[1] - true_az) == pytest.approx(front_law, abs=2e-3 * AS)
            assert wrap(back[1] - np.pi - true_az) == pytest.approx(back_law, abs=2e-3 * AS)


# -- sensitivities -------------------------------------------------------------------------------------------
def test_point_jacobian_of_a_perfect_tracker_is_the_inverse_of_the_spherical_jacobian():
    x = spherical_to_cartesian(3.0, 0.5, 0.3)
    expected = np.linalg.inv(spherical_jacobian(*cartesian_to_spherical(x)))
    np.testing.assert_allclose(observation_point_jacobian(x, ZERO), expected, rtol=1e-6, atol=1e-9)


def test_parameter_jacobian_of_a_perfect_tracker_is_what_the_equations_say():
    d, az, el = 3.0, 0.5, 0.3
    x = spherical_to_cartesian(d, az, el)
    jacobian = observation_parameter_jacobian(x, ZERO)
    assert jacobian[0, INDEX["lambda_m"]] == pytest.approx(1.0, abs=1e-9)
    assert jacobian[0, INDEX["mu"]] == pytest.approx(d, rel=1e-9)
    assert jacobian[1, INDEX["a_A1_rad"]] == pytest.approx(np.cos(az), rel=1e-6)
    assert jacobian[1, INDEX["b_A2_rad"]] == pytest.approx(np.sin(2 * az), rel=1e-6)
    assert jacobian[2, INDEX["a_E0_rad"]] == pytest.approx(1.0, abs=1e-9)
    assert jacobian[2, INDEX["b_E1_rad"]] == pytest.approx(np.sin(el), rel=1e-6)
    assert jacobian[1, INDEX["gamma_rad"]] == pytest.approx(-1.0 / np.cos(el), rel=1e-6)   # the single-face laws above
    assert jacobian[1, INDEX["alpha_rad"]] == pytest.approx(np.tan(el), rel=1e-6)
    assert jacobian[0, INDEX["a_A1_rad"]] == pytest.approx(0.0, abs=1e-6)
    assert jacobian.shape == (3, 16)


@pytest.mark.parametrize("face", [FRONT, BACK])
def test_parameter_jacobian_does_not_depend_on_the_step_chosen(face):
    """Halving and doubling the per-parameter steps changes the Jacobian by
    under 1e-6 of its largest entry in each row (the justification of
    `PARAMETER_STEPS`)."""
    x = spherical_to_cartesian(4.0, 1.1, np.deg2rad(35.0))
    reference = observation_parameter_jacobian(x, HUGHES, face)
    for factor in (0.5, 2.0):
        other = observation_parameter_jacobian(x, HUGHES, face, steps=PARAMETER_STEPS * factor)
        for row in range(3):
            assert np.abs(other[row] - reference[row]).max() < 1e-6 * np.abs(reference[row]).max()


# -- batch versions (for the network layers) --------------------------------------------------------------
def random_shots(k=120, seed=0, high=77.0):
    rng = np.random.default_rng(seed)
    points = np.array([spherical_to_cartesian(rng.uniform(0.3, 6.4), rng.uniform(-np.pi, np.pi),
                                              np.deg2rad(rng.uniform(-60, high))) for _ in range(k)])
    return points, rng.choice([FRONT, BACK], k)


def test_batch_forward_model_equals_the_scalar_one():
    rng = np.random.default_rng(1)
    d, theta, phi = rng.uniform(0.3, 6.4, 40), rng.uniform(-np.pi, np.pi, 40), rng.uniform(-1.0, 4.0, 40)
    origin, direction = beam_geometry_batch(HUGHES.values, theta, phi)
    points = point_from_true_batch(HUGHES.values, d, theta, phi)
    readings = readings_from_true_batch(HUGHES.values, d, theta, phi)
    for i in range(40):
        o, n = beam_geometry(HUGHES.values, theta[i], phi[i])
        np.testing.assert_allclose(origin[i], o, atol=1e-15)
        np.testing.assert_allclose(direction[i], n, atol=1e-15)
        np.testing.assert_allclose(points[i], point_from_true(HUGHES.values, d[i], theta[i], phi[i]), atol=1e-14)
        np.testing.assert_allclose(readings[i], readings_from_true(HUGHES.values, d[i], theta[i], phi[i]), atol=1e-14)


def test_batch_solve_and_readings_equal_the_scalar_ones_and_one_vector_per_shot_works():
    points, faces = random_shots()
    batch = true_to_observed_batch(points, HUGHES.values, faces)
    for i in range(len(points)):
        np.testing.assert_allclose(batch[i], true_to_observed(points[i], HUGHES, faces[i]), atol=1e-12)
    per_shot = np.where(np.arange(len(points))[:, None] % 2 == 0, HUGHES.values, 0.0)  # every other shot from a perfect tracker
    mixed = true_to_observed_batch(points, per_shot, faces)
    for i in range(0, len(points), 2):
        np.testing.assert_allclose(mixed[i], batch[i], atol=1e-12)
    for i in range(1, len(points), 2):
        np.testing.assert_allclose(mixed[i], true_to_observed(points[i], ZERO, faces[i]), atol=1e-12)


def test_batch_solve_handles_the_beam_pointing_straight_up_and_reports_the_blind_spot():
    points = np.array([point_from_true(HUGHES.values, 3.0, theta, np.pi / 2) for theta in (-2.0, 0.3, 2.5)])
    q = solve_true_coordinates_batch(points, HUGHES.values, [FRONT] * 3)
    np.testing.assert_allclose(point_from_true_batch(HUGHES.values, *q.T), points, atol=1e-12)
    with pytest.raises(RuntimeError, match="vertical axis"):
        solve_true_coordinates_batch(np.array([[0.0, 0.0, 3.0]]), HUGHES.values, [FRONT])


def test_sensitivities_agree_with_finite_differences_through_the_solve():
    points, faces = random_shots(12, seed=3)
    j_point, j_params, p_shift = reading_sensitivities(points, HUGHES.values, faces)
    for i in range(len(points)):
        fd_point = observation_point_jacobian(points[i], HUGHES, faces[i])
        fd_params = observation_parameter_jacobian(points[i], HUGHES, faces[i])
        assert np.abs(j_point[i] - fd_point).max() < 1e-7 * np.abs(fd_point).max()
        assert np.abs(j_params[i] - fd_params).max() < 1e-7 * np.abs(fd_params).max()
        np.testing.assert_allclose(p_shift[i], np.linalg.solve(fd_point, fd_params), rtol=1e-6, atol=1e-6 * np.abs(p_shift[i]).max())


def test_point_shift_of_a_perfect_tracker_is_the_spherical_jacobian_times_the_reading_error_minus_the_beam_shift():
    """P = F_q G_q^-1 G_c - F_c. For an encoder error (G_c only) the reported
    point moves through the spherical Jacobian: a_E,0 of 1 arcsec moves it by d
    along the elevation direction. For a pure beam offset b_y,0 (F_c only) it
    moves by minus the offset."""
    d, az, el = 3.0, 0.5, 0.3
    x = spherical_to_cartesian(d, az, el)[None, :]
    _, _, shift = reading_sensitivities(x, ZERO.values)
    elevation_direction = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    np.testing.assert_allclose(shift[0][:, INDEX["a_E0_rad"]], d * elevation_direction, rtol=1e-6)
    np.testing.assert_allclose(shift[0][:, INDEX["b_y0_m"]], -np.array([-np.sin(az), np.cos(az), 0.0]), atol=1e-6)
    np.testing.assert_allclose(shift[0][:, INDEX["lambda_m"]], np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]), rtol=1e-6)
