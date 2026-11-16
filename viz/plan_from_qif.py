"""Plan the tracker stations for a part or assembly described by a QIF file.

    python -m viz.plan_from_qif scenes/qif/SheetMetal_QIF_Results_6_samples.QIF
    python -m viz.plan_from_qif part.qif --list                 # what the file contains, and stop
    python -m viz.plan_from_qif assembly.qif --scope 3 --stations 2 3 4 5
    python -m viz.plan_from_qif part.qif --stl part.stl --stl-metres-per-unit 0.001

Reads the file (`interchange.qif`), builds the cell round the part
(`planning.qif_cell.build_qif_cell`: what is planned, what is skipped and why,
every assumption), searches station layouts for each objective and station
count (`plan_qif_cell`), and writes

    results/qif_plan_<name>.json      everything, including the notes
    results/qif_plan_<name>.png/.svg  plan view of the cell and the stations

`--stl` gives the part's shape (in the file's coordinates) so it can block
sight lines; without it only range and incidence angle limit what a station
sees. Nothing here computes: it reads, calls `src/`, prints and draws.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import re
from pathlib import Path

import numpy as np

from geometry.mesh import TriangleMesh, load_stl
from interchange.qif import read_qif
from planning.qif_cell import DEFAULT_DE, build_qif_cell, plan_qif_cell

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")


def draw_plan(cell, plan, n_stations, path_stem):
    """Plan view: the track, the part's extent, the reference nests, every
    planned target (marker shape = its cluster) and each objective's stations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    objectives = [o for o in plan["objectives"] if plan["plans"].get(f"{o}|{n_stations}", {}).get("feasible")]
    if not objectives:
        return None
    fig, axes = plt.subplots(1, len(objectives), figsize=(6.4 * len(objectives), 5.6), squeeze=False)
    track = cell.track
    s = np.linspace(0.0, track.perimeter_m, 400)
    path = np.array([track.position_xy_m(v) for v in s])
    points = cell.scenario.target_points_m
    cluster_names = [c.name for c in cell.clusters]
    for ax, name in zip(axes[0], objectives):
        entry = plan["plans"][f"{name}|{n_stations}"]
        ax.plot(path[:, 0], path[:, 1], color="0.55", lw=1.0, ls="--", label="station track")
        low, high = cell.envelope_min_m, cell.envelope_max_m
        ax.add_patch(plt.Rectangle((low[0], low[1]), high[0] - low[0], high[1] - low[1], fill=False, ec="k", lw=1.2,
                                   label="extent of located features"))
        nests = points[cell.n_part_targets:]
        if len(nests):
            ax.scatter(nests[:, 0], nests[:, 1], marker="s", s=22, facecolors="none", edgecolors="0.35", label="reference nests")
        for k, cluster in enumerate(cluster_names):
            idx = [p.target_index for p in cell.planned if p.cluster == cluster]
            ax.scatter(points[idx, 0], points[idx, 1], marker=MARKERS[k % len(MARKERS)], s=26, label=cluster, zorder=3)
        for i, station in enumerate(entry["stations"]):
            x, y, z = station["position_cell_m"]
            ax.scatter([x], [y], marker="*", s=260, color="k", zorder=4, label="tracker station" if i == 0 else None)
            ax.annotate(f"S{i + 1}\n{z:.1f} m high", (x, y), textcoords="offset points", xytext=(9, 7), fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlabel("cell x (m)")
        ax.set_ylabel("cell y (m)")
        ax.set_title(f"objective {name}: {n_stations} stations\n"
                     f"expected wrong decisions {entry['total_risk']:.3g}, worst TUR {entry['worst_tur']:.1f}:1, "
                     f"largest u {max(entry['u_um'].values()):.1f} um", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.suptitle("Plan view, true scale (part markers overlap when the part is small against the cell)", fontsize=9)
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(str(path_stem) + suffix, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path_stem) + ".png"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("qif", type=Path)
    parser.add_argument("--list", action="store_true", help="print what the file contains and stop")
    parser.add_argument("--scope", default=None, help="id of the part or assembly to plan (see --list); default: the root")
    parser.add_argument("--stations", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--objectives", nargs="+", default=["A1", "risk"], choices=["A0", "A1", "TUR", "risk", "max_u"])
    parser.add_argument("--rule", choices=["simple", "best"], default="simple")
    parser.add_argument("--alignment", choices=["auto", "datums", "best_fit"], default="auto")
    parser.add_argument("--up-axis", choices=["x", "y", "z"], default="z", help="which axis of the file is vertical in the cell")
    parser.add_argument("--stl", type=Path, default=None, help="the part's shape, in the file's coordinates, to block sight lines")
    parser.add_argument("--stl-metres-per-unit", type=float, default=None, help="default: the QIF file's own unit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_DE["maxiter"])
    parser.add_argument("--popsize", type=int, default=DEFAULT_DE["popsize"])
    args = parser.parse_args()

    doc = read_qif(args.qif)
    print(f"{args.qif.name}: QIF {doc.version}, unit {doc.linear_unit_name}, {len(doc.features)} features, "
          f"{len(doc.characteristics)} characteristics, {doc.n_inspections} earlier inspections")
    for scope_id, label in doc.scope_names().items():
        print(f"  scope {scope_id}: {label} ({len(doc.characteristics_in(scope_id))} characteristics)")
    if args.list:
        kinds = {}
        for c in doc.characteristics.values():
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        print("  characteristic types:", ", ".join(f"{k} x{n}" for k, n in sorted(kinds.items())))
        for warning in dict.fromkeys(doc.warnings):
            print("  note:", warning)
        return

    obstacle = None
    if args.stl is not None:
        mesh = load_stl(args.stl)
        scale = args.stl_metres_per_unit if args.stl_metres_per_unit is not None else doc.metres_per_unit
        obstacle = TriangleMesh(mesh.vertices_m * scale, mesh.faces)
    cell = build_qif_cell(doc, scope_id=args.scope, up_axis=args.up_axis, obstacle_part_frame=obstacle, alignment=args.alignment)
    summary = cell.summary()
    print(f"planned {len(cell.planned)} characteristics ({sum(p.approximate for p in cell.planned)} approximate), "
          f"skipped {len(cell.skipped)}")
    reasons = {}
    for _, reason in cell.skipped:
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in reasons.items():
        print(f"  skipped x{count}: {reason}")
    print(f"cell: part extent {np.round(summary['part_extent_m'], 2).tolist()} m, track {summary['track']['perimeter_m']:.1f} m round, "
          f"{summary['n_reference_nests']} reference nests; part frame: {cell.alignment}")
    for cluster in cell.clusters:
        print(f"  cluster {cluster.name}: {len(cluster.members)} characteristics, mean {cluster.mean_m * 1e6:+.0f} um, "
              f"std {cluster.std_m * 1e6:.0f} um; {cluster.source}")
    for note in cell.notes:
        print("  note:", note)

    de = {**DEFAULT_DE, "maxiter": args.maxiter, "popsize": args.popsize}
    plan = plan_qif_cell(cell, args.stations, args.objectives, args.rule, args.seed, de)
    plan["source_file"] = args.qif.name
    print(f"expected wrong decisions if nothing is measured: {plan['risk_if_nothing_is_measured']:.3g}")
    print(f"{'stations':>8} {'objective':>9} {'wrong decisions':>16} {'worst TUR':>10} {'largest u (um)':>15} {'shots':>6}")
    for k in args.stations:
        for name in args.objectives:
            entry = plan["plans"][f"{name}|{k}"]
            if not entry["feasible"]:
                print(f"{k:>8} {name:>9} {'no feasible layout found':>16}")
                continue
            print(f"{k:>8} {name:>9} {entry['total_risk']:>16.4g} {entry['worst_tur']:>9.1f}:1 "
                  f"{max(entry['u_um'].values()):>15.1f} {entry['n_shots']:>6}")
    stem = RESULTS_DIR / ("qif_plan_" + re.sub(r"[^A-Za-z0-9]+", "_", args.qif.stem).strip("_").lower()
                          + (f"_scope_{args.scope}" if args.scope else ""))
    stem.with_suffix(".json").write_text(json.dumps(plan, indent=1, default=float))
    print("wrote", stem.with_suffix(".json"))
    feasible = [k for k in args.stations if any(plan["plans"][f"{o}|{k}"]["feasible"] for o in args.objectives)]
    if feasible:
        print("wrote", draw_plan(cell, plan, feasible[0], stem))


if __name__ == "__main__":
    main()
