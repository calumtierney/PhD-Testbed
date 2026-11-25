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
this is the picture of the same thing, CLAUDE.md §8). One small panel per
parameter, each with its own title and law and its own vertical scale (the
signatures differ by a factor of a hundred), so nothing needs a legend:

  top row     azimuth two-face difference, (azimuth_back - pi) - azimuth_front
  bottom row  elevation two-face sum, elevation_front + elevation_back - pi

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

# (parameter, title, closed-form law as TeX, law in radians as a function of elevation)
PANEL_A = [
    ("gamma_rad", "γ  beam not square\nto transit axis", r"$+2\gamma\,/\cos\varphi$",
     lambda el: 2 * HUGHES.values[INDEX["gamma_rad"]] / np.cos(el)),
    ("alpha_rad", "α  transit axis\ntilted", r"$-2\alpha\,\tan\varphi$",
     lambda el: -2 * HUGHES.values[INDEX["alpha_rad"]] * np.tan(el)),
    ("b_y0_m", "b_y,0  beam offset\nsideways", r"$+2b_{y0}\,/\,(d\cos\varphi)$",
     lambda el: 2 * HUGHES.values[INDEX["b_y0_m"]] / (RANGE_M * np.cos(el))),
    ("a_A1_rad", "a_A,1  azimuth encoder,\ncos θ", r"$-2a_{A1}\cos\theta$",
     lambda el: -2 * HUGHES.values[INDEX["a_A1_rad"]] * np.cos(AZIMUTH_RAD) + 0.0 * el),
    ("b_A1_rad", "b_A,1  azimuth encoder,\nsin θ", r"$-2b_{A1}\sin\theta$",
     lambda el: -2 * HUGHES.values[INDEX["b_A1_rad"]] * np.sin(AZIMUTH_RAD) + 0.0 * el),
]
PANEL_B = [
    ("e_x_m", "e_x  transit axis\noffset", r"$+2e_x\sin\varphi\,/\,d$",
     lambda el: 2 * HUGHES.values[INDEX["e_x_m"]] * np.sin(el) / RANGE_M),
    ("b_z0_m", "b_z,0  beam offset\nvertically", r"$-2b_{z0}\,/\,d$",
     lambda el: -2 * HUGHES.values[INDEX["b_z0_m"]] / RANGE_M + 0.0 * el),
    ("a_E0_rad", "a_E,0  elevation\nzero offset", r"$+2a_{E0}$",
     lambda el: 2 * HUGHES.values[INDEX["a_E0_rad"]] + 0.0 * el),
    ("b_E1_rad", "b_E,1  elevation encoder,\nsin φ", r"$+2b_{E1}\sin\varphi$",
     lambda el: 2 * HUGHES.values[INDEX["b_E1_rad"]] * np.sin(el)),
    ("a_E2_rad", "a_E,2  elevation encoder,\ncos 2φ", r"$+2a_{E2}\cos 2\varphi$",
     lambda el: 2 * HUGHES.values[INDEX["a_E2_rad"]] * np.cos(2 * el)),
]
ROW_COLOURS = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2 of the reference palette: one colour per row


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


def draw_panel(ax, entry, which: int, colour: str, show_ylabel: bool, ylabel: str):
    name, title, law_tex, law = entry
    elevation = np.deg2rad(ELEVATIONS_DEG)
    curve = signatures(name)[which]
    assert np.max(np.abs(curve - law(elevation) / AS)) < 5e-3, name  # the dotted law must lie under the curve
    ax.plot(ELEVATIONS_DEG, curve, color=colour, lw=2.2, marker="o", markevery=13, ms=5.5, markerfacecolor=colour,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.plot(ELEVATIONS_DEG, law(elevation) / AS, color=INK, ls=":", lw=1.2, zorder=4)
    ax.axhline(0.0, color=MUTED, lw=0.8, zorder=1)
    ax.set_title(title, color=INK, fontsize=9.5, loc="left", pad=22)
    ax.text(0.0, 1.03, "law: " + law_tex, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
    ax.set_xlim(-60, 77)
    pad = 0.12 * max(float(np.ptp(curve)), 0.4)
    ax.set_ylim(min(curve.min(), 0.0) - pad, max(curve.max(), 0.0) + pad)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if show_ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)


def main():
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 7.6))
    for column, entry in enumerate(PANEL_A):
        draw_panel(axes[0, column], entry, 0, ROW_COLOURS[0], column == 0, "azimuth two-face difference (arcsec)")
    for column, entry in enumerate(PANEL_B):
        draw_panel(axes[1, column], entry, 1, ROW_COLOURS[1], column == 0, "elevation two-face sum (arcsec)")
        axes[1, column].set_xlabel("target elevation (degrees)", color=INK, fontsize=9)
    fig.suptitle(f"Two-face signature of each NPL parameter alone, at its Hughes et al. 2011 value "
                 f"(target at {RANGE_M:g} m, azimuth {AZIMUTH_RAD:g} rad)", color=INK, fontsize=12, y=0.995)
    fig.text(0.5, 0.945, "Top row: (azimuth back − π) − azimuth front.   Bottom row: elevation front + elevation back − π.   "
             "Solid with markers: the model.   Dotted: the closed-form law from tests/test_npl_model.py.   "
             "Each panel has its own vertical scale.", ha="center", color=MUTED, fontsize=9)
    fig.text(0.5, 0.005,
             "Parameters with no signature at all, not drawn: range offset λ, range scale μ, the second azimuth harmonic a_A,2 and b_A,2, a_E,1 and b_E,2. "
             "A two-face test cannot see them; the network test can.",
             ha="center", va="bottom", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93), h_pad=2.2)
    RESULTS_DIR.mkdir(exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(RESULTS_DIR / f"two_face_signatures{suffix}", dpi=200, bbox_inches="tight")
    print("wrote", RESULTS_DIR / "two_face_signatures.png")


if __name__ == "__main__":
    main()
