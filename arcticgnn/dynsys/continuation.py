"""Numerical continuation of periodic states of the seasonally forced model.

The forced system has no steady state.  Its analogue is a periodic annual cycle,
which is a fixed point of the map that advances the model by exactly one year,

.. math::

    G(x, p) \\;=\\; \\Phi_{1\\,\\mathrm{yr}}(x, p) - x \\;=\\; 0 .

Continuation is carried out on that map.  Stability follows from the Floquet
multipliers, the eigenvalues of the monodromy matrix
:math:`\\partial \\Phi / \\partial x`; the cycle is stable when all of them lie
inside the unit circle, and a saddle-node bifurcation of cycles occurs where one
of them leaves it through :math:`+1`.

Pseudo-arclength is used instead of stepping the parameter directly.  Stepping
the parameter cannot pass a fold, because the branch turns back on itself there
and the solver either fails or jumps to the other stable branch; the unstable
middle segment joining the two, which is what proves the branches belong to one
S-shaped curve, would never be found.  Arclength continuation follows the curve
through the turn and recovers it.

Folds located from the geometry of the branch are refined by solving the
extended system

.. math::

    G(x, p) = 0, \\qquad G_x(x, p)\\, v = 0, \\qquad v^{\\mathsf T} v - 1 = 0,

whose solution has a Jacobian with an exact zero eigenvalue, equivalently a
Floquet multiplier of exactly one.

Every Jacobian here is built by finite differences of a one-year integration.
That is affordable only because the model integrates a batch of columns at the
cost of one, so all the perturbed evaluations needed for a Jacobian are packed
into a single call.  It also requires the integrator to use a fixed step:
adaptive step-size control makes the map a slightly discontinuous function of
its argument, and differencing that returns noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Problem interface
# --------------------------------------------------------------------------- #


class PeriodicProblem:
    """A one-year map together with the scales needed to continue it.

    Subclasses supply :meth:`advance`, which must accept a batch of states and a
    matching batch of control parameter values and return the batch advanced by
    one period.
    """

    #: Number of state variables.
    ndim: int = 2
    #: Characteristic variation of each state variable.  The arclength norm
    #: mixes state and parameter, so without scaling the component with the
    #: largest raw magnitude dominates the tangent and the corrector steps far
    #: enough to land on a different branch.
    state_scale: np.ndarray = np.ones(2)
    #: Name of the control parameter, used only for labelling.
    parameter: str = "parameter"

    def advance(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Advance a batch of states by one period."""
        raise NotImplementedError

    def observable(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Scalar diagnostic plotted against the control parameter."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

@dataclass
class Branch:
    """A continued branch of periodic states."""

    parameter: str
    p: np.ndarray = field(default_factory=lambda: np.empty(0))
    x: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    stable: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    observable: np.ndarray = field(default_factory=lambda: np.empty(0))
    multiplier: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __len__(self) -> int:
        return int(self.p.size)

    def segment(self, stable: bool) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(p, observable)`` on the stable or the unstable portion."""
        mask = self.stable if stable else ~self.stable
        return self.p[mask], self.observable[mask]


@dataclass
class Fold:
    """A saddle-node bifurcation of periodic states."""

    parameter: str
    p: float
    x: np.ndarray
    observable: float
    direction: str            # 'upper_to_lower' or 'lower_to_upper'
    residual: float
    refined: bool


@dataclass
class Hysteresis:
    """A bistable window bounded by two folds."""

    parameter: str
    p_forward: float          # fold crossed as the parameter increases
    p_reverse: float          # fold crossed as it decreases
    width: float
    jump: float               # discontinuity in the observable at the fold
    folds: tuple


# --------------------------------------------------------------------------- #
# Scaled system
# --------------------------------------------------------------------------- #

class _Scaled:
    """``G(y, w) = 0`` in scaled coordinates, with batched derivatives."""

    def __init__(self, problem: PeriodicProblem, p_scale: float = 1.0):
        self.problem = problem
        self.n = problem.ndim
        self.scale = np.asarray(problem.state_scale, float)
        self.p_scale = float(p_scale) if p_scale else 1.0

    def to_y(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, float) / self.scale

    def to_x(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, float) * self.scale

    def to_w(self, p: float) -> float:
        return float(p) / self.p_scale

    def to_p(self, w: float) -> float:
        return float(w) * self.p_scale

    # -- residual ------------------------------------------------------------ #

    def G_batch(self, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Scaled residual for a batch of points."""
        x = self.to_x(np.atleast_2d(y))
        p = np.atleast_1d(np.asarray(w, float)) * self.p_scale
        return (self.to_y(self.problem.advance(x, p))
                - np.atleast_2d(np.asarray(y, float)))

    def G(self, y: np.ndarray, w: float) -> np.ndarray:
        return self.G_batch(np.atleast_2d(y), np.array([w]))[0]

    def derivatives(self, y: np.ndarray, w: float, eps: float = 1e-5
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(G, dG/dy, dG/dw)`` from one batched evaluation.

        The base point and every perturbed point are stacked into a single call,
        so the whole Jacobian costs one integration.
        """
        n = self.n
        y = np.asarray(y, float)
        points = [y]
        params = [w]
        for j in range(n):
            step = eps * max(abs(y[j]), 1.0)
            shifted = y.copy()
            shifted[j] += step
            points.append(shifted)
            params.append(w)
        step_w = eps * max(abs(w), 1.0)
        points.append(y.copy())
        params.append(w + step_w)

        values = self.G_batch(np.array(points), np.array(params))

        base = values[0]
        jac = np.empty((n, n))
        for j in range(n):
            step = eps * max(abs(y[j]), 1.0)
            jac[:, j] = (values[1 + j] - base) / step
        dw = (values[n + 1] - base) / step_w
        return base, jac, dw

    def monodromy(self, y: np.ndarray, w: float, eps: float = 1e-5
                  ) -> np.ndarray:
        """Monodromy matrix in scaled coordinates."""
        _, jac, _ = self.derivatives(y, w, eps)
        return jac + np.eye(self.n)

    def solve(self, y0: np.ndarray, w: float, tol: float = 1e-11,
              max_iter: int = 40) -> np.ndarray:
        """Newton solve for a fixed point at fixed parameter."""
        y = np.asarray(y0, float).copy()
        for _ in range(max_iter):
            base, jac, _ = self.derivatives(y, w)
            if np.linalg.norm(base) < tol:
                return y
            try:
                y = y - np.linalg.solve(jac, base)
            except np.linalg.LinAlgError:
                break
        if np.linalg.norm(self.G(y, w)) > 1e-7:
            raise RuntimeError(
                f"fixed point did not converge at {self.problem.parameter}"
                f"={self.to_p(w):g}")
        return y


# --------------------------------------------------------------------------- #
# Pseudo-arclength continuation
# --------------------------------------------------------------------------- #

#: A corrected step further than this multiple of the arclength step has landed
#: on a different branch and is rejected.
JUMP_TOLERANCE = 3.0


def _tangent(sys: _Scaled, y: np.ndarray, w: float,
             previous: np.ndarray | None) -> np.ndarray:
    """Unit tangent to the solution curve, oriented to follow ``previous``."""
    n = sys.n
    _, jac, dw = sys.derivatives(y, w)

    matrix = np.zeros((n + 1, n + 1))
    matrix[:n, :n] = jac
    matrix[:n, n] = dw
    matrix[n, :] = previous if previous is not None else np.eye(n + 1)[n]

    rhs = np.zeros(n + 1)
    rhs[n] = 1.0
    try:
        tangent = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        tangent = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    norm = np.linalg.norm(tangent)
    if norm == 0.0 or not np.isfinite(norm):
        raise RuntimeError("degenerate tangent")
    tangent = tangent / norm
    if previous is not None and float(tangent @ previous) < 0.0:
        tangent = -tangent
    return tangent


def _corrector(sys: _Scaled, predicted: np.ndarray, previous: np.ndarray,
               tangent: np.ndarray, ds: float, max_iter: int = 12,
               tol: float = 1e-10) -> np.ndarray | None:
    """Newton correction back onto the curve within the arclength plane."""
    n = sys.n
    z = predicted.copy()
    residual = np.full(n + 1, np.inf)

    for _ in range(max_iter):
        y, w = z[:n], float(z[n])
        base, jac, dw = sys.derivatives(y, w)

        residual = np.empty(n + 1)
        residual[:n] = base
        residual[n] = float(tangent @ (z - previous)) - ds
        if np.linalg.norm(residual) < tol:
            return z

        matrix = np.zeros((n + 1, n + 1))
        matrix[:n, :n] = jac
        matrix[:n, n] = dw
        matrix[n, :] = tangent
        try:
            step = np.linalg.solve(matrix, residual)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None
        z = z - step

    return z if np.linalg.norm(residual) < 1e-7 else None


def continue_branch(problem: PeriodicProblem, x0: np.ndarray, p0: float,
                    p_min: float, p_max: float, ds: float = 0.02,
                    ds_min: float = 1e-6, ds_max: float = 0.08,
                    max_steps: int = 4000, direction: int = 1,
                    p_scale: float | None = None) -> Branch:
    """Trace one branch of periodic states by pseudo-arclength continuation.

    Parameters
    ----------
    problem
        The one-year map to continue.
    x0, p0
        A converged periodic state and the parameter value it belongs to.
    p_min, p_max
        Continuation stops once the parameter leaves this interval.
    ds
        Initial arclength step in scaled units, adapted within
        ``[ds_min, ds_max]``.
    direction
        ``+1`` to set off towards increasing parameter, ``-1`` for decreasing.
    """
    if p_scale is None:
        p_scale = max(abs(p_max - p_min) / 20.0, 1e-6)

    sys = _Scaled(problem, p_scale=p_scale)
    n = sys.n
    w0 = sys.to_w(p0)
    y = sys.solve(sys.to_y(x0), w0)
    z = np.concatenate([y, [w0]])

    tangent = _tangent(sys, y, w0, None)
    if np.sign(tangent[n]) != np.sign(direction):
        tangent = -tangent

    ws, ys = [w0], [y.copy()]
    step = ds

    for _ in range(max_steps):
        corrected = _corrector(sys, z + step * tangent, z, tangent, step)
        if (corrected is not None
                and np.linalg.norm(corrected - z) > JUMP_TOLERANCE * step):
            corrected = None

        if corrected is None:
            step *= 0.5
            if step < ds_min:
                break
            continue

        y_new, w_new = corrected[:n], float(corrected[n])
        try:
            tangent = _tangent(sys, y_new, w_new, tangent)
        except RuntimeError:
            break
        z = corrected
        ws.append(w_new)
        ys.append(y_new.copy())

        if not (p_min <= sys.to_p(w_new) <= p_max):
            break
        step = min(step * 1.1, ds_max)

    ps = np.array([sys.to_p(w) for w in ws])
    xs = np.array([sys.to_x(y) for y in ys])
    return _finalise(sys, ps, xs)


def _finalise(sys: _Scaled, ps: np.ndarray, xs: np.ndarray) -> Branch:
    """Attach stability and the plotted diagnostic to a traced branch."""
    count = ps.size
    stable = np.empty(count, bool)
    multiplier = np.empty(count)

    for i in range(count):
        eigenvalues = np.linalg.eigvals(
            sys.monodromy(sys.to_y(xs[i]), sys.to_w(ps[i])))
        multiplier[i] = np.abs(eigenvalues).max()
        stable[i] = multiplier[i] < 1.0

    observable = sys.problem.observable(xs, ps)
    return Branch(parameter=sys.problem.parameter, p=ps, x=xs, stable=stable,
                  observable=np.asarray(observable, float),
                  multiplier=multiplier)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

def _refine_fold(sys: _Scaled, y0: np.ndarray, w0: float
                 ) -> tuple[np.ndarray, float, float, bool]:
    """Newton-refine a fold through the extended system."""
    from scipy.optimize import root

    n = sys.n
    _, jac, _ = sys.derivatives(y0, w0)
    values, vectors = np.linalg.eig(jac)
    v0 = np.real(vectors[:, int(np.argmin(np.abs(values)))])
    norm = np.linalg.norm(v0)
    v0 = v0 / norm if norm > 0 else np.ones(n) / np.sqrt(n)

    def extended(u):
        y, v, w = u[:n], u[n:2 * n], float(u[2 * n])
        base, jac_local, _ = sys.derivatives(y, w)
        return np.concatenate([base, jac_local @ v, [v @ v - 1.0]])

    solution = root(extended, np.concatenate([y0, v0, [w0]]), method="hybr",
                    tol=1e-9, options={"eps": 1e-4})
    if not solution.success:
        return y0, w0, float(np.linalg.norm(sys.G(y0, w0))), False

    y = solution.x[:n]
    w = float(solution.x[2 * n])
    return y, w, float(np.linalg.norm(sys.G(y, w))), True


def find_folds(problem: PeriodicProblem, branch: Branch,
               p_scale: float | None = None) -> list[Fold]:
    """Locate saddle-node bifurcations along a continued branch.

    Folds are found where the branch reverses direction in the control
    parameter, which is where a Floquet multiplier passes through one.
    """
    if len(branch) < 3:
        return []
    if p_scale is None:
        span = float(branch.p.max() - branch.p.min())
        p_scale = max(span / 20.0, 1e-6)

    sys = _Scaled(problem, p_scale=p_scale)
    steps = np.diff(branch.p)
    turns = np.where(np.sign(steps[:-1]) * np.sign(steps[1:]) < 0)[0] + 1

    folds: list[Fold] = []
    for i in turns:
        y, w, residual, ok = _refine_fold(sys, sys.to_y(branch.x[i]),
                                          sys.to_w(branch.p[i]))
        p = sys.to_p(w)
        x = sys.to_x(y)
        value = float(np.atleast_1d(
            problem.observable(x[None, :], np.array([p])))[0])

        low, high = max(i - 5, 0), min(i + 6, len(branch))
        window = branch.observable[low:high]
        midpoint = 0.5 * (float(window.max()) + float(window.min()))
        direction = "lower_to_upper" if value < midpoint else "upper_to_lower"

        folds.append(Fold(parameter=branch.parameter, p=p, x=x,
                          observable=value, direction=direction,
                          residual=residual, refined=ok))
    return folds


def hysteresis(problem: PeriodicProblem, branch: Branch) -> Hysteresis | None:
    """Quantify the bistable window enclosed by a pair of folds.

    Returns ``None`` when fewer than two folds were found, meaning the branch is
    monostable across the interval that was continued.
    """
    folds = find_folds(problem, branch)
    if len(folds) < 2:
        return None

    ordered = sorted(folds, key=lambda f: f.p)
    low, high = ordered[0], ordered[-1]

    inside = (branch.p > low.p) & (branch.p < high.p)
    jump = (float(branch.observable[inside].max()
                  - branch.observable[inside].min())
            if inside.any() else abs(high.observable - low.observable))

    return Hysteresis(parameter=branch.parameter, p_forward=high.p,
                      p_reverse=low.p, width=abs(high.p - low.p), jump=jump,
                      folds=tuple(ordered))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def bifurcation_diagram(problem: PeriodicProblem, x0: np.ndarray, p_start: float,
                        p_min: float, p_max: float, ds: float = 0.02,
                        max_steps: int = 4000
                        ) -> tuple[Branch, list[Fold], Hysteresis | None]:
    """Continue in both directions from ``p_start`` and analyse the result.

    The two halves are joined so that a single branch spans the whole S-shaped
    curve, including the unstable segment that connects the two stable states.
    """
    p_scale = max(abs(p_max - p_min) / 20.0, 1e-6)
    backward = continue_branch(problem, x0, p_start, p_min, p_max, ds=ds,
                               max_steps=max_steps, direction=-1,
                               p_scale=p_scale)
    forward = continue_branch(problem, x0, p_start, p_min, p_max, ds=ds,
                              max_steps=max_steps, direction=+1,
                              p_scale=p_scale)

    merged = Branch(
        parameter=problem.parameter,
        p=np.concatenate([backward.p[::-1], forward.p[1:]]),
        x=np.concatenate([backward.x[::-1], forward.x[1:]]),
        stable=np.concatenate([backward.stable[::-1], forward.stable[1:]]),
        observable=np.concatenate([backward.observable[::-1],
                                   forward.observable[1:]]),
        multiplier=np.concatenate([backward.multiplier[::-1],
                                   forward.multiplier[1:]]),
    )
    return merged, find_folds(problem, merged, p_scale), hysteresis(problem, merged)
