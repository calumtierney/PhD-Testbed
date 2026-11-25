"""Two-face signatures of the NPL geometric parameters: the check that the
model's signs and frame are right.

    python -m viz.make_two_face_figure

A perfect laser tracker gives the same point from its front and back face. A
real one does not, and each parameter leaves a different fingerprint in the
disagreement, as a function of the elevation of the target. This draws those
fingerprints from `instruments.npl_model` -- one parameter at a time, at its
Hughes et al. (2011) value, target at 3 m and azimuth 0.7 rad -- and lays the
closed-form laws derived in `tests/test_npl_model.py::test_two_face_laws` over
them as dotted lines. If a sign, a rotation or a transposed frame were wrong
the curves would not sit on the dotted laws (the tests would fail first;
this is the picture of the same thing, CLAUDE.md §8).

  (a) azimuth two-face difference, (azimuth_back - pi) - azimuth_front
  (b) elevation two-face sum, elevation_front + elevation_back - pi

Parameters not drawn have exactly zero signature in both: the range offset
and scale (lambda, mu), the second azimuth harmonic (a_A2, b_A2), and a_E1
and b_E2 (their contributions cancel between faces). Those are the
parameters a two-face test cannot see and the network test must.
Writes results/two_face_signatures.png and .svg.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry.spherical import spherical_to_cartesian
from instruments.npl_model import ARCSEC_TO_RAD, BACK, FRONT, INDEX, NPLErrorParameters, true_to_observed

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
HUGHES = NPLErrorParameters.hughes_2011_uncompensated()
RANGE_M, AZIMUTH_RAD = 3.0, 0.7
ELEVATIONS_DEG = np.linspace(-60.0, 77.0, 138)  # the tracker's mechanical range
AS = ARCSEC_TO_RAD
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"
# categorical slots 1-5 of the reference palette, plus a marker and a line style each, so colour is never alone
STYLES = [("#2a78d6", "o", "-"), ("#eb6834", "s", "--"), ("#1baf7a", "^", "-."), ("#eda100", "D", (0, (5, 1))), ("#e87ba4", "v", (0, (1, 1)))]

# panel (a): (parameter, label, closed-form law in radians as a function of elevation)
PANEL_A = [
    ("gamma_rad", "γ  beam not square to transit axis", lambda el: 2 * HUGHES.values[INDEX["gamma_rad"]] / np.cos(el)),
    ("alpha_rad", "α  transit axis tilted", lambda el: -2 * HUGHES.values[INDEX["alpha_rad"]] * np.tan(el)),
    ("b_y0_m", "b_y,0  beam offset sideways", lambda el: 2 * HUGHES.values[INDEX["b_y0_m"]] / (RANGE_M * np.cos(el))),
    ("a_A1_rad", "a_A,1  azimuth encoder, cos", lambda el: -2 * HUGHES.values[INDEX["a_A1_rad"]] * np.cos(AZIMUTH_RAD) + 0.0 * el),
    ("b_A1_rad", "b_A,1  azimuth encoder, sin", lambda el: -2 * HUGHES.values[INDEX["b_A1_rad"]] * np.sin(AZIMUTH_RAD) + 0.0 * el),
]
# panel (b)
PANEL_B = [
    ("e_x_m", "e_x  transit axis offset", lambda el: 2 * HUGHES.values[INDEX["e_x_m"]] * np.sin(el) / RANGE_M),
    ("b_z0_m", "b_z,0  beam offset vertically", lambda el: -2 * HUGHES.values[INDEX["b_z0_m"]] / RANGE_M + 0.0 * el),
    ("a_E0_rad", "a_E,0  elevation zero offset", lambda el: 2 * HUGHES.values[INDEX["a_E0_rad"]] + 0.0 * el),
    ("b_E1_rad", "b_E,1  elevation encoder, sin φ", lambda el: 2 * HUGHES.values[INDEX["b_E1_rad"]] * np.sin(el)),
    ("a_E2_rad", "a_E,2  elevation encoder, cos 2φ", lambda el: 2 * HUGHES.values[INDEX["a_E2_rad"]] * np.cos(2 * el)),
]


def one_parameter(name: str) -> NPLErrorParameters:
    values = np.zeros_like(HUGHES.values)
    values[INDEX[name]] = HUGHES.values[INDEX[name]]
    return NPLErrorParameters(values, np.zeros_like(HUGHES.covariance))


def signatures(name: str):
    """(azimuth difference, elevation sum) in arcseconds along `ELEVATIONS_DEG`."""
    params = one_parameter(name)
    difference, total = [], []
    for el_deg in ELEVATIONS_DEG:
        x = spherical_to_cartesian(RANGE_M, AZIMUTH_RAD, np.deg2rad(el_deg))
        front, back = true_to_observed(x, params, FRONT), true_to_observed(x, params, BACK)
        d = (back[1] - np.pi) - front[1]
        difference.append(((d + np.pi) % (2 * np.pi) - np.pi) / AS)
        total.append((front[2] + back[2] - np.pi) / AS)
    return np.array(difference), np.array(total)


def draw(ax, entries, which: int, title: str, ylabel: str):
    elevation = np.deg2rad(ELEVATIONS_DEG)
    for (name, label, law), (colour, marker, line) in zip(entries, STYLES):
        curve = signatures(name)[which]
        ax.plot(ELEVATIONS_DEG, curve, color=colour, ls=line, lw=1.8, marker=marker, markevery=11, ms=6,
                markerfacecolor=colour, markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
        ax.plot(ELEVATIONS_DEG, law(elevation) / AS, color=INK, ls=":", lw=1.0, zorder=4)
        assert np.max(np.abs(curve - law(elevation) / AS)) < 5e-3, name  # the dotted law must lie under the curve
    ax.plot([], [], color=INK, ls=":", lw=1.0, label="closed-form law (test_two_face_laws)")
    ax.axhline(0.0, color=MUTED, lw=0.8, zorder=1)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("target elevation (degrees)", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xlim(-60, 77)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
    draw(axes[0], PANEL_A, 0, "(a) azimuth two-face difference", "(azimuth back − π) − azimuth front (arcseconds)")
    draw(axes[1], PANEL_B, 1, "(b) elevation two-face sum", "elevation front + elevation back − π (arcseconds)")
    fig.suptitle(f"Two-face signature of each NPL parameter alone, at its Hughes et al. 2011 value; target at "
                 f"{RANGE_M:g} m, azimuth {AZIMUTH_RAD:g} rad", color=INK, fontsize=11.5)
    fig.text(0.5, 0.005,
             "No signature at all (not drawn): range offset λ, range scale μ, second azimuth harmonic a_A,2 and b_A,2, "
             "a_E,1, b_E,2. A two-face test cannot see them; the network test can.\n"
             "γ falls as 1/cos φ and α as tan φ with opposite sign; b_y,0 and e_x, b_z,0 shrink with range, γ and α do not.",
             ha="center", va="bottom", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    RESULTS_DIR.mkdir(exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(RESULTS_DIR / f"two_face_signatures{suffix}", dpi=200, bbox_inches="tight")
    print("wrote", RESULTS_DIR / "two_face_signatures.png")


if __name__ == "__main__":
    main()
