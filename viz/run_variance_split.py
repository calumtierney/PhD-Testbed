"""Gate 6 of the NPL-model task: the paper's 1000-point variance split.

    python -m viz.run_variance_split

Writes results/npl_variance_split.json: the share of the total coordinate
variance of 1000 points (1-5 m, random azimuth and elevation within the
mechanical -60..+77 degrees, half on each face) from the angle sensors, the range
sensor and each geometric parameter, with `hughes_2011_residual()`; and, as an
exploration only (no input is changed), how the overall share moves if the
correlations the paper mentions but does not quantify were present.
"""
import json
from pathlib import Path

import numpy as np

from instruments.npl_model import INDEX, NPLErrorParameters
from network.variance_split import parameter_variance_split

RESULTS = Path(__file__).resolve().parent.parent / "results" / "npl_variance_split.json"


def main():
    residual = NPLErrorParameters.hughes_2011_residual()
    sigma = residual.standard_uncertainties()
    out = {"paper": {"angle_noise": 0.80, "parameters": 0.20, "per_parameter_range": [0.001, 0.038]}, "runs": {}}
    for distribution in ("uniform_angle", "uniform_direction"):
        out["runs"][distribution] = {str(seed): parameter_variance_split(residual, seed=seed, distribution=distribution) for seed in range(3)}
    out["correlation_what_if"] = {}
    for label, pairs in (("a_E0-a_E2", [("a_E0_rad", "a_E2_rad")]), ("gamma-b_y0 and a_E0-b_z0", [("gamma_rad", "b_y0_m"), ("a_E0_rad", "b_z0_m")])):
        for rho in (-0.9, -0.5, 0.5, 0.9):
            covariance = residual.covariance.copy()
            for a, b in pairs:
                i, j = INDEX[a], INDEX[b]
                covariance[i, j] = covariance[j, i] = rho * sigma[i] * sigma[j]
            out["correlation_what_if"][f"{label} rho={rho:+.1f}"] = parameter_variance_split(NPLErrorParameters(residual.values, covariance))["parameters"]
    RESULTS.write_text(json.dumps(out, indent=1))
    base = out["runs"]["uniform_angle"]["0"]
    print(f"parameters {100 * base['parameters']:.1f}% (paper ~20%), angle noise {100 * base['angle_noise']:.1f}% (paper ~80%), "
          f"range noise {100 * base['range_noise']:.2f}%")
    for name, share in sorted(base["per_parameter"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:10s} {100 * share:5.2f}%")
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
