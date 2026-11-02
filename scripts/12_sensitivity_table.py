"""Write the supplementary table reporting how the bistable window responds to the
prescribed parameters.

Reads ``runs/sensitivity.json`` from the sensitivity stage and emits
``manuscript/si/tableS3_sensitivity.tex``.  The table answers one question: which
of the quantities that were prescribed from observations, and never fitted, change
the width of the bistable window enough to matter.  Every row is one such quantity
held at the low and the high end of a plausible range, with the surface flux offset
recalibrated at each so that the perturbed configuration still reproduces the same
observed annual mean thickness as the control.

Kept separate from the figure scripts because it produces no figure.
"""

from __future__ import annotations

import argparse
import json

from arcticgnn.config import FIGURE_DIR, RUNS_DIR, ensure_dirs


HEADER = r"""\begin{table}[!htbp]
\centering
\caption{\textbf{Response of the bistable window to the parameters that were
prescribed.} The control configuration has an albedo transition scale of
\SI{%(scale).2f}{\metre} and a window of
\SI{%(control).2f}{\watt\per\metre\squared}. Each parameter below is moved to the
low and the high end of a plausible range, the one free quantity of the surface
energy budget is recalibrated at each value so that the perturbed configuration
still reproduces the observed annual mean thickness of the central Arctic, and the
branch of annual cycles is continued again. A dash in the window column marks a
configuration that no surface energy budget reconciles with the observed
thickness, which is itself a constraint on the parameter. The window is the
interval of greenhouse forcing over which perennial and seasonally ice free states
coexist.}
\label{tab:sensitivity}
\small
\begin{tabular}{lcccc}
\toprule
Quantity & Value & Flux offset & Window & Threshold \\
 & & (\si{\watt\per\metre\squared}) & (\si{\watt\per\metre\squared}) &
 (\si{\watt\per\metre\squared}) \\
\midrule
"""

FOOTER = r"""\bottomrule
\end{tabular}
\end{table}
"""


def row(label: str, unit: str, case: dict, default: float) -> str:
    """One LaTeX row for one perturbed value."""
    shown = f"{case['value']:g}" + (f" {unit}" if unit and unit != "1" else "")
    if not case["feasible"]:
        return (f"{label} & {shown} & -- & -- & -- \\\\")
    threshold = ("--" if case["threshold"] is None
                 else f"{case['threshold']:.2f}")
    return (f"{label} & {shown} & {case['longwave_offset']:.2f} & "
            f"{case['width']:.2f} & {threshold} \\\\")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    arguments = parser.parse_args()

    ensure_dirs()
    path = RUNS_DIR / "sensitivity.json"
    if not path.exists():
        raise SystemExit("run scripts/11_sensitivity.py first")
    data = json.loads(path.read_text())

    lines = [HEADER % {"scale": data["albedo_scale"],
                       "control": data["control"]["width"]}]

    # The control row first, so the comparison the table is built around is the
    # one the reader sees before any perturbation.
    control = data["control"]
    lines.append(f"Control & -- & {control['longwave_offset']:.2f} & "
                 f"{control['width']:.2f} & "
                 + ("--" if control["threshold"] is None
                    else f"{control['threshold']:.2f}")
                 + " \\\\")
    lines.append(r"\midrule")

    for entry in data["parameters"]:
        label, unit = entry["label"], entry["unit"]
        default = entry["default"]
        for case in entry["cases"]:
            mark = " (default)" if abs(case["value"] - default) < 1e-9 else ""
            lines.append(row(label + mark, unit, case, default))

    lines.append(FOOTER)

    out = FIGURE_DIR.parent / "si" / "tableS3_sensitivity.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")

    summary = data.get("summary")
    if summary:
        print(f"window across every perturbation: "
              f"{summary['width_min']:.2f} to {summary['width_max']:.2f} W m-2")
        print(f"bistable throughout: {summary['bistable_in_all_cases']}")


if __name__ == "__main__":
    main()
