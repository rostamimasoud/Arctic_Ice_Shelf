"""Figure styling for the manuscript.

The categorical palette below was checked with a colour-vision-deficiency
validator: in this order every adjacent pair clears a deutan/tritan separation
of dE >= 8 (OKLab x100) and a normal-vision separation of dE >= 20.  The order is
therefore fixed -- assign hues by position and never cycle or reshuffle them,
because reordering reintroduces the pairs that fail.

Three of the hues sit below 3:1 contrast against a white surface, so every chart
that uses them must carry a legend or direct labels; identity is never conveyed
by colour alone.  That is the house rule here in any case.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #

#: Validated categorical order.  Do not permute.
CATEGORICAL: tuple[str, ...] = (
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # bluish green
    "#E69F00",   # orange
    "#CC79A7",   # reddish purple
    "#56B4E9",   # sky blue
)

#: Circulation regimes.  These are identities, not magnitudes.
REGIME = {
    "cold": "#0072B2",
    "warm": "#D55E00",
    "unstable": "#8C8C8C",
    "bistable": "#E8E4DC",
}

#: Model / data provenance.
SOURCE = {
    "box_model": "#0072B2",
    "emulator": "#D55E00",
    "nemo": "#009E73",
    "roms_utas": "#E69F00",
    "mitgcm": "#CC79A7",
    "observations": "#3A3A3A",
}

#: Ink tokens.  Text never wears a series colour.
INK = {"primary": "#1A1A1A", "secondary": "#4A4A4A", "muted": "#7A7A7A",
       "grid": "#DCDCDC", "surface": "#FFFFFF"}

#: Sequential ramp for magnitude: a single hue, light to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "arcticgnn_seq", ["#EAF3F8", "#B8D8EA", "#7FB8D8", "#3D8FBF", "#0072B2", "#004E7A"])

#: Diverging ramp for signed anomalies: two hues about a neutral grey midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "arcticgnn_div", ["#00485F", "#0072B2", "#8FBED5", "#EFEFEF",
                   "#E9A87C", "#D55E00", "#8A3D00"])

#: Melt-rate ramp: perceptually ordered, light for refreezing through to dark for
#: intense melt.  Registered so it can be referenced by name.
MELT = LinearSegmentedColormap.from_list(
    "arcticgnn_melt", ["#F7F7F5", "#CFE3EC", "#8FC0D6", "#E8C07A", "#D55E00", "#7A2E00"])

def _register(cmap) -> None:
    """Register a colormap across matplotlib versions, ignoring re-registration."""
    try:
        mpl.colormaps.register(cmap)                       # matplotlib >= 3.6
        return
    except AttributeError:
        pass
    except ValueError:
        return                                             # already registered
    try:
        mpl.cm.register_cmap(name=cmap.name, cmap=cmap)    # matplotlib < 3.6
    except (AttributeError, ValueError):
        pass


for _cm in (SEQUENTIAL, DIVERGING, MELT):
    _register(_cm)


# --------------------------------------------------------------------------- #
# Nature-style rcParams
# --------------------------------------------------------------------------- #

#: Nature column widths in inches.
WIDTH_SINGLE = 3.50    # 89 mm
WIDTH_ONE_HALF = 4.72  # 120 mm
WIDTH_DOUBLE = 7.20    # 183 mm

RC = {
    "figure.dpi": 150,
    # Nature asks 300 dpi for halftone, 600 for combination art, 1000 or more
    # for line work.  Vector output is unaffected by this; it governs the raster
    # copies written alongside for convenience.
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
    "savefig.facecolor": INK["surface"],
    "savefig.transparent": False,
    "figure.facecolor": INK["surface"],
    "axes.facecolor": INK["surface"],
    "figure.constrained_layout.use": False,

    # Nimbus Sans is metrically compatible with Helvetica, which is the face
    # Nature sets.  The fallbacks are the equivalent clones so that the figures
    # render identically on a machine without it, and DejaVu is last because its
    # wide, low-contrast letterforms are what mark a figure as untouched
    # matplotlib output.
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial",
                        "Liberation Sans", "FreeSans", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.title_fontsize": 6.5,
    "figure.titlesize": 8,
    # Typographic minus rather than a hyphen, which is a different glyph and
    # noticeably too short in front of a numeral.
    "axes.unicode_minus": True,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 4),

    # A recessive frame: only the two axes the reader actually reads against.
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "axes.edgecolor": INK["secondary"],
    "axes.labelcolor": INK["primary"],
    "axes.labelpad": 2.5,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "axes.titlepad": 3.5,
    "axes.axisbelow": True,

    # Ticks point outward, are short, and carry a minor set at half weight, so
    # that the reader can interpolate without the frame becoming a ladder.
    "xtick.color": INK["secondary"],
    "ytick.color": INK["secondary"],
    "xtick.labelcolor": INK["primary"],
    "ytick.labelcolor": INK["primary"],
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.35,
    "ytick.minor.width": 0.35,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    "grid.color": INK["grid"],
    "grid.linewidth": 0.35,
    "grid.alpha": 1.0,
    "axes.grid": False,

    # Thin marks with rounded joins, which reads better at column width than
    # the default mitre and avoids the spiky corners of a dense line.
    "lines.linewidth": 1.1,
    "lines.markersize": 3.0,
    "lines.markeredgewidth": 0.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.dash_capstyle": "round",
    "lines.dashed_pattern": (4.0, 1.6),
    "lines.dotted_pattern": (1.0, 1.4),
    "lines.antialiased": True,
    "patch.linewidth": 0.4,
    "patch.antialiased": True,
    "hatch.linewidth": 0.4,

    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.45,
    "legend.columnspacing": 1.1,
    "legend.labelspacing": 0.28,
    "legend.borderpad": 0.25,
    "legend.borderaxespad": 0.4,
    "legend.markerscale": 1.0,

    "axes.prop_cycle": mpl.cycler(color=list(CATEGORICAL)),
    "errorbar.capsize": 1.2,
    "image.interpolation": "nearest",
    "image.origin": "lower",

    # Embed as TrueType so that the text in the vector output stays selectable
    # and editable by the typesetter, and never as Type 3, which many journals
    # reject outright.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 6,
    "svg.fonttype": "none",
    "text.usetex": False,
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
}


def use_style() -> None:
    """Apply the manuscript rcParams globally.

    Keys unknown to the installed matplotlib are skipped rather than raising, so
    the same figures build on an older local install and on the cluster.
    """
    known = {k: v for k, v in RC.items() if k in mpl.rcParams}
    mpl.rcParams.update(known)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def figure(width: float = WIDTH_DOUBLE, height: float | None = None,
           nrows: int = 1, ncols: int = 1, **kwargs):
    """Create a styled figure sized to a Nature column width."""
    use_style()
    height = height if height is not None else width * 0.62
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.06) -> None:
    """Place a bold panel letter above and left of the axes.

    The letter is set in the figure's own face at a size slightly above the body
    text, which is the convention, and is deliberately not given a background
    patch: a patch behind a label is a sign of a plot whose layout was never
    settled.
    """
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="left", color=INK["primary"],
            clip_on=False)


#: A spine is only cut back when its outermost ticks sit within this fraction of
#: the view span from the ends of the axis.
TRIM_TOLERANCE = 0.06


def trim_spines(ax, axis: str = "both") -> None:
    """Shorten each spine to the span of its ticks, where that is safe to do.

    A frame running the full width of the axes implies data reaching its ends,
    which is rarely true, so cutting the spine back to the first and last tick
    reads as considered rather than default.  The trim is skipped when the
    outermost ticks fall well inside the view limits: the spine would then stop
    short of data that is plotted beyond it, and of the minor ticks that
    :func:`use_style` draws there, which looks like a rendering fault rather
    than a choice.
    """
    for name, get_ticks, get_limits in (
            ("bottom", ax.get_xticks, ax.get_xlim),
            ("left", ax.get_yticks, ax.get_ylim)):
        if axis not in ("both", "x" if name == "bottom" else "y"):
            continue
        low, high = get_limits()
        span = high - low
        if span <= 0:
            continue
        inside = [t for t in get_ticks() if low <= t <= high]
        if len(inside) < 2:
            continue
        first, last = min(inside), max(inside)
        if (first - low) / span > TRIM_TOLERANCE:
            continue
        if (high - last) / span > TRIM_TOLERANCE:
            continue
        ax.spines[name].set_bounds(first, last)


def bare_axes(ax) -> None:
    """Strip a map axis of frame, ticks and minor ticks.

    Map panels carry no numeric scale, so the tick machinery that
    :func:`use_style` switches on globally has to be removed again here; left in
    place it produces stray marks along the edges of every map.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    for spine in ax.spines.values():
        spine.set_visible(False)


def soften_grid(ax, axis: str = "y") -> None:
    """Add a recessive grid behind the data.

    Only one direction is drawn by default. A full grid in both directions turns
    the panel into graph paper and competes with the marks it is meant to help
    the reader place.
    """
    ax.grid(True, axis=axis, which="major", color=INK["grid"], linewidth=0.35,
            zorder=0)
    ax.set_axisbelow(True)


def direct_label(ax, x: float, y: float, text: str, color: str,
                 dx: float = 3.0, **kwargs) -> None:
    """Label a series at its end.

    The marker beside the text carries the identity; the text itself stays in
    ink, per the house rule that text never wears a series colour.
    """
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", ha="left", fontsize=6.5, color=INK["primary"], **kwargs)
    ax.plot([x], [y], marker="o", markersize=3.0, color=color, zorder=5,
            markeredgecolor=INK["surface"], markeredgewidth=0.5)


def categorical(n: int) -> list[str]:
    """First ``n`` categorical hues, in the validated order.

    Raises
    ------
    ValueError
        If more than six are requested.  A seventh generated hue would not have
        been validated; group the tail into 'Other' or use small multiples.
    """
    if n > len(CATEGORICAL):
        raise ValueError(
            f"{n} categorical hues requested but only {len(CATEGORICAL)} are validated; "
            f"group the remainder or switch to small multiples")
    return list(CATEGORICAL[:n])


def symmetric_norm(values: Sequence[float] | np.ndarray, quantile: float = 0.99):
    """A diverging normaliser centred on zero, robust to outliers."""
    v = np.abs(np.asarray(values, float))
    v = v[np.isfinite(v)]
    lim = float(np.quantile(v, quantile)) if v.size else 1.0
    return mpl.colors.Normalize(vmin=-lim, vmax=lim)


def polish(fig) -> None:
    """Apply the finishing passes that every figure in the study receives.

    Two things happen. Spines on axes carrying a numeric scale are cut back to
    the span of their ticks, and minor ticks are removed from any axis whose
    major ticks were cleared, which is how the map panels are built. Doing this
    once at save time keeps the treatment uniform across figures instead of
    depending on each panel remembering to ask for it.
    """
    for ax in fig.get_axes():
        # A colour bar is an axes too, and its own styling is set elsewhere.
        if getattr(ax, "_colorbar", None) is not None:
            continue
        has_x = len(ax.get_xticks()) > 0
        has_y = len(ax.get_yticks()) > 0
        if not has_x:
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        if not has_y:
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        if has_x and ax.spines["bottom"].get_visible():
            trim_spines(ax, axis="x")
        if has_y and ax.spines["left"].get_visible():
            trim_spines(ax, axis="y")


def save(fig, path, formats: Sequence[str] = ("pdf", "png")) -> list[str]:
    """Polish and save a figure in several formats, returning the paths written."""
    from pathlib import Path

    polish(fig)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt)
        written.append(str(out))
    return written
