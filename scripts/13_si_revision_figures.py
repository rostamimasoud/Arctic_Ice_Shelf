"""Stage 13: supplementary figures for the revision.

Two figures that answer questions raised in review and that the earlier
supplementary set does not cover.

S6 shows how the width of the bistable window responds to the quantities that
were prescribed from observations and never fitted.  Every perturbed
configuration is recalibrated so that it reproduces the same observed annual mean
thickness as the control, which is what makes the comparison like for like: a
change in the window is attributable to the parameter and not to a different
control climate.

S7 reports the early warning experiment across the full set of noise
realisations.  The earlier figure follows one realisation in detail; this one
reports how often each indicator warned, which is the quantity a reader needs in
order to judge whether such an indicator would be of use.

Conventions match the other figures: no in-image titles, bold lower case panel
labels, axis labels carrying units, and colour never the sole distinguishing
feature.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import FIGURE_DIR, RUNS_DIR, ensure_dirs
from arcticgnn.viz import style


def load(name: str) -> dict | None:
    path = RUNS_DIR / name
    if not path.exists():
        print(f"  {name} is missing, skipping the figures that need it")
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# S6: sensitivity of the window to the prescribed parameters
# --------------------------------------------------------------------------- #

#: Short forms of the parameter names.  The table carries the full wording; a
#: figure axis has to stay legible at column width.
SHORT = {"Mixed layer depth": "Mixed layer depth",
         "Sub-halocline layer depth": "Deep layer depth",
         "Ventilation of the deep layer": "Deep ventilation",
         "Mixing under closed ice": "Mixing under ice",
         "Additional mixing over open water": "Mixing, open water",
         "Albedo over ice": "Albedo, ice",
         "Albedo over open water": "Albedo, open water",
         "Background ocean heat flux": "Ocean heat flux"}


def figure_sensitivity(data: dict) -> None:
    """What the window does when the prescribed quantities are moved."""
    import matplotlib.pyplot as plt

    style.use_style()
    figure, axes_grid = plt.subplots(1, 3, figsize=(style.WIDTH_DOUBLE, 3.0))
    # The two left panels carry parameter names on the vertical axis, so they
    # need room that the default margins do not leave.
    figure.subplots_adjust(left=0.135, right=0.985, wspace=0.95, bottom=0.30)

    control = float(data["control"]["width"])
    entries = [e for e in data["parameters"] if e.get("width_range")]

    # (a) the window reached at each end of every parameter range, against the
    # control.  A horizontal layout keeps the parameter names readable.  A
    # parameter with one feasible end is kept and drawn as the single point it
    # is, since dropping it would hide a configuration the observations exclude.
    axes = axes_grid[0]
    labels, lows, highs, single = [], [], [], []
    for entry in entries:
        feasible = [c for c in entry["cases"] if c["feasible"]]
        if not feasible:
            continue
        labels.append(SHORT.get(entry["label"], entry["label"]))
        lows.append(float(min(c["width"] for c in feasible)))
        highs.append(float(max(c["width"] for c in feasible)))
        single.append(len(feasible) < len(entry["cases"]))

    y = np.arange(len(labels))
    axes.hlines(y, lows, highs, color=style.CATEGORICAL[0], linewidth=2.4,
                zorder=2)
    axes.plot(lows, y, linestyle="none", marker="o", markersize=3.4,
              color=style.CATEGORICAL[0], zorder=3, label="Range spanned")
    axes.plot(highs, y, linestyle="none", marker="o", markersize=3.4,
              color=style.CATEGORICAL[0], zorder=3)
    # An open marker flags a parameter one of whose ends no budget reaches.
    partial = np.array(single, bool)
    if partial.any():
        axes.plot(np.array(highs)[partial], y[partial], linestyle="none",
                  marker="o", markersize=6.0, markerfacecolor="none",
                  markeredgecolor=style.REGIME["warm"], markeredgewidth=1.0,
                  zorder=4, label="One end excluded")
    axes.axvline(control, color=style.INK["secondary"], linewidth=0.9,
                 linestyle="--", zorder=1, label="Control")
    axes.set_yticks(y)
    axes.set_yticklabels(labels, fontsize=5.4)
    axes.set_xlabel("Bistable window (W m$^{-2}$)")
    axes.set_xlim(0, max(highs) * 1.12)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1,
                fontsize=5.2, frameon=False)
    style.soften_grid(axes, axis="x")
    style.panel_label(axes, "a")

    # (b) the same as a departure from the control, which is the quantity the
    # robustness claim rests on.
    axes = axes_grid[1]
    departure = [max(abs(lo - control), abs(hi - control))
                 for lo, hi in zip(lows, highs)]
    order = np.argsort(departure)
    axes.barh(np.arange(len(order)), [departure[i] for i in order],
              color=style.CATEGORICAL[2], height=0.62)
    axes.set_yticks(np.arange(len(order)))
    axes.set_yticklabels([labels[i] for i in order], fontsize=5.4)
    axes.set_xlabel("Largest change in window (W m$^{-2}$)")
    style.soften_grid(axes, axis="x")
    style.panel_label(axes, "b")

    # (c) the calibrated offset each perturbation required, which shows that the
    # control climate was held fixed by refitting rather than left to drift.
    axes = axes_grid[2]
    offsets, widths, marks = [], [], []
    for entry in entries:
        for case in entry["cases"]:
            if case["feasible"]:
                offsets.append(float(case["longwave_offset"]))
                widths.append(float(case["width"]))
                marks.append(entry["label"])
    axes.plot(offsets, widths, linestyle="none", marker="o", markersize=3.2,
              color=style.CATEGORICAL[0], alpha=0.85)
    axes.axhline(control, color=style.INK["secondary"], linewidth=0.9,
                 linestyle="--")
    axes.annotate("Control window", xy=(min(offsets), control),
                  xytext=(2, 4), textcoords="offset points", fontsize=5.8,
                  color=style.INK["secondary"], va="bottom")
    axes.set_xlabel("Recalibrated flux offset (W m$^{-2}$)")
    axes.set_ylabel("Bistable window (W m$^{-2}$)")
    style.soften_grid(axes)
    style.panel_label(axes, "c")

    style.save(figure, FIGURE_DIR / "figureS6_sensitivity")
    print("  figure S6 written")


# --------------------------------------------------------------------------- #
# S7: early warning across every noise realisation
# --------------------------------------------------------------------------- #

NAMES = {"variance": "Variance",
         "autocorrelation": "Lag one autocorrelation",
         "recovery_rate": "Recovery rate",
         "skewness": "Skewness",
         "irreversibility": "Time irreversibility"}

#: Short forms for the horizontal axis of panel b, where the full names collide.
BRIEF = {"variance": "Variance",
         "autocorrelation": "Autocorr.",
         "recovery_rate": "Recovery",
         "skewness": "Skewness",
         "irreversibility": "Irrevers."}


def figure_warning_ensemble(warning: dict) -> None:
    """How often each indicator warned, across the whole set of realisations."""
    import matplotlib.pyplot as plt

    style.use_style()
    figure, axes_grid = plt.subplots(1, 3, figsize=(style.WIDTH_DOUBLE, 3.0))
    figure.subplots_adjust(wspace=0.44, bottom=0.30)

    members = warning["members"]
    n = len(members)
    order = [k for k in NAMES if k in warning["detection_rate"]]

    # (a) detection rate per indicator, with the binomial interval that the
    # sample size supports, so the reader can see what n buys.
    axes = axes_grid[0]
    rates = np.array([warning["detection_rate"][k] for k in order])
    error = 1.96 * np.sqrt(np.clip(rates * (1 - rates), 0, None) / max(n, 1))
    y = np.arange(len(order))
    axes.barh(y, rates, color=style.CATEGORICAL[0], height=0.58, zorder=2)
    axes.errorbar(rates, y, xerr=error, linestyle="none", capsize=2.0,
                  elinewidth=0.9, color=style.INK["primary"], zorder=3)
    axes.set_yticks(y)
    axes.set_yticklabels([NAMES[k] for k in order], fontsize=5.8)
    axes.set_xlim(0, 1)
    axes.set_xlabel(f"Fraction of {n} realisations that warned")
    style.soften_grid(axes, axis="x")
    style.panel_label(axes, "a")

    # (b) the trend statistic itself for every realisation, so that the spread
    # behind each rate in (a) is visible rather than summarised away.
    axes = axes_grid[1]
    for i, key in enumerate(order):
        taus = [m["indicators"]["trends"][key]["tau"] for m in members
                if key in m["indicators"]["trends"]]
        jitter = (np.arange(len(taus)) % 7 - 3) * 0.028
        axes.plot(np.full(len(taus), i) + jitter, taus, linestyle="none",
                  marker="o", markersize=2.0, alpha=0.5,
                  color=style.CATEGORICAL[0])
        axes.plot([i - 0.26, i + 0.26], [np.median(taus)] * 2,
                  color=style.CATEGORICAL[1], linewidth=1.6, zorder=3)
    axes.axhline(0.0, color=style.INK["muted"], linewidth=0.8, linestyle=":")
    axes.set_xticks(np.arange(len(order)))
    axes.set_xticklabels([BRIEF[k] for k in order], fontsize=5.6,
                         rotation=30, ha="right")
    axes.set_ylabel("Kendall rank correlation with time")
    # The median rule is labelled below the cloud, where it cannot sit on top
    # of the points it summarises.
    axes.annotate("Median", xy=(len(order) - 1.0, axes.get_ylim()[0]),
                  xytext=(0, 6), textcoords="offset points", fontsize=5.8,
                  color=style.CATEGORICAL[1], ha="center", va="bottom")
    style.soften_grid(axes)
    style.panel_label(axes, "b")

    # (c) how the estimate of the detection rate settles as realisations are
    # added, which is the direct answer to whether the sample is large enough.
    axes = axes_grid[2]
    # The autocorrelation and the recovery rate are one monotone function of
    # each other, so their trend tests agree realisation by realisation and the
    # two curves coincide.  They are drawn with different dashes and widths so
    # that the coincidence is visible as such and not as a missing curve.
    dashes = [(None, None), (None, None), (3.0, 1.6)]
    widths = [1.1, 2.2, 1.1]
    for i, key in enumerate(order[:3]):
        flags = np.array([bool(m["indicators"]["trends"][key]["significant"])
                          for m in members], float)
        running = np.cumsum(flags) / np.arange(1, len(flags) + 1)
        line, = axes.plot(np.arange(1, len(flags) + 1), running,
                          color=style.CATEGORICAL[i], linewidth=widths[i],
                          label=NAMES[key])
        if dashes[i][0] is not None:
            line.set_dashes(list(dashes[i]))
    axes.set_xlabel("Realisations included")
    axes.set_ylabel("Running detection rate")
    axes.set_ylim(0, 1)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
                fontsize=5.5, frameon=False)
    style.soften_grid(axes)
    style.panel_label(axes, "c")

    style.save(figure, FIGURE_DIR / "figureS7_warning_ensemble")
    print("  figure S7 written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default=None,
                        help="sensitivity | warning")
    arguments = parser.parse_args()

    ensure_dirs()
    if arguments.only in (None, "sensitivity"):
        data = load("sensitivity.json")
        if data:
            figure_sensitivity(data)
    if arguments.only in (None, "warning"):
        warning = load("warning.json")
        if warning:
            figure_warning_ensemble(warning)
    print(f"supplementary figures are in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
