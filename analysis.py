"""Shared model results and conservative checks beyond the upstream engine."""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from adl_theory import adl_dynamic_effects

MAX_HORIZON = 500
UNSTABLE_HORIZON = 30
TOLERANCE = 0.01
MILESTONE_SHARES = (50, 90, 95, 99)
ROOT_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Results:
    table: pd.DataFrame
    inverse_roots: np.ndarray
    classification: str
    effective_orders: tuple[int, int]
    long_run: float | None
    mean: float | None
    halfway: int | None
    milestones: dict[int, int | None] | None
    timing: str
    settling: int | None
    band: float | None
    near_zero: bool
    overshoots: bool
    oscillates: bool
    truncated: bool

    @property
    def crossing_99(self):
        return self.milestones[99] if self.milestones is not None else None

    @property
    def horizon(self):
        return len(self.table) - 1


def _first_crossing(shares, target):
    indices = np.flatnonzero(shares >= target)
    return int(indices[0]) if len(indices) else None


def _tail_inside(phi, errors, band):
    """Bound the infinite tail by finding a contracting companion matrix power.

    For A^k with infinity norm < 1, future states can be grouped into
    blocks of k. The largest norm over the first k powers bounds every
    subsequent block, including intermediate lags.
    """
    if not len(phi):
        return True
    order = len(phi)
    matrix = np.eye(order, k=-1)
    matrix[0] = phi
    power = np.eye(order)
    largest = 1.0
    state_size = np.max(np.abs(errors[-order:]))
    for _ in range(MAX_HORIZON):
        power = power @ matrix
        norm = np.linalg.norm(power, ord=np.inf)
        if norm < 1 - 1e-12:
            return largest * state_size < band * (1 - 1e-10)
        largest = max(largest, norm)
    return False


def analyze(phi, beta, horizon=None):
    """Evaluate one specification, with a shared slice for every display."""
    phi = np.asarray(phi, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if (phi.ndim != 1 or beta.ndim != 1 or len(phi) > 2 or not 1 <= len(beta) <= 3
            or not np.all(np.isfinite(phi)) or not np.all(np.isfinite(beta))):
        raise ValueError('Enter finite coefficients for orders between 0 and 2.')
    if horizon is not None and (not isinstance(horizon, (int, np.integer))
                                or not 0 <= horizon <= MAX_HORIZON):
        raise ValueError(f'Enter a whole-number horizon between 0 and {MAX_HORIZON}.')

    phi = np.trim_zeros(phi, 'b')
    beta = np.trim_zeros(beta, 'b') if np.any(beta) else np.zeros(1)
    # An effective AR order of zero has no finite inverse roots.
    with warnings.catch_warnings(), np.errstate(over='ignore', invalid='ignore'):
        warnings.filterwarnings('ignore', message='The ADL model is dynamically unstable.')
        engine = adl_dynamic_effects(phi, beta, horizon=MAX_HORIZON)
    summary = engine['summary']
    roots = engine['stability']['inverse_roots']
    radius = engine['stability']['max_inverse_root']
    if radius > 1 + ROOT_TOLERANCE:
        classification = 'Explosive'
    elif radius >= 1 - ROOT_TOLERANCE:
        classification = 'Unit-circle root (within numerical tolerance)'
    else:
        classification = 'Stable'
    table = engine['table']
    finite = np.isfinite(table[['Dynamic Effect', 'Cumulative Effect']]).all(axis=1)
    end = int(np.flatnonzero(~finite)[0]) if not finite.all() else len(table)
    table = table.iloc[:end].copy()
    dynamic = table['Dynamic Effect'].to_numpy()
    cumulative = table['Cumulative Effect'].to_numpy()

    stable = classification == 'Stable'
    long_run = summary['long_run_effect']
    long_run = float(long_run) if stable and np.isfinite(long_run) else None
    # A scale floor covers the all-zero response. Peak observed effects and
    # input coefficients keep the absolute band meaningful under cancellation.
    scale = max(1.0, float(np.max(np.abs(beta))), float(np.max(np.abs(cumulative))))
    near_zero = long_run is not None and abs(long_run) <= 1e-8 * scale
    band = None
    settling = None
    mean = halfway = None
    milestones = None
    timing = 'Unavailable'
    if long_run is not None:
        band = TOLERANCE * (scale if near_zero else abs(long_run))
        errors = cumulative - long_run
        if end == MAX_HORIZON + 1 and _tail_inside(phi, errors, band):
            outside = np.flatnonzero(np.abs(errors) > band)
            settling = int(outside[-1] + 1) if len(outside) else 0

    if stable and np.isfinite(summary['mean_lag']):
        mean = float(summary['mean_lag'])
    if stable and np.isfinite(summary['median_lag']):
        halfway = int(summary['median_lag'])
    if long_run is not None and not np.isclose(long_run, 0.0):
        with np.errstate(over='ignore'):
            shares = cumulative / long_run
        milestones = {50: halfway}
        milestones.update({percent: _first_crossing(shares, percent / 100)
                           for percent in MILESTONE_SHARES[1:]})
    if long_run is not None and long_run != 0:
        if np.any(dynamic * np.sign(long_run) < 0):
            timing = 'Signed'
        elif not len(phi) or (np.all(phi >= 0) and end == MAX_HORIZON + 1):
            timing = 'Distribution'
        else:
            timing = 'Sign not certified'
    if not stable:
        table[['% Total Effect', '% Cumulative Effect']] = np.nan

    selected = horizon
    if selected is None:
        selected = (settling if settling is not None else MAX_HORIZON) if stable else UNSTABLE_HORIZON
    shown = table.iloc[:selected + 1]
    if long_run is not None and long_run != 0:
        oriented_errors = (shown['Cumulative Effect'].to_numpy() - long_run) * np.sign(long_run)
        significant = np.sign(oriented_errors[np.abs(oriented_errors) > 1e-12 * abs(long_run)])
        overshoots = bool(np.any(significant > 0))
        oscillates = bool(np.count_nonzero(np.diff(significant)) >= 2)
    else:
        overshoots = False
        significant = np.sign(dynamic[:selected + 1][np.abs(dynamic[:selected + 1]) > 1e-12 * scale])
        oscillates = bool(np.count_nonzero(np.diff(significant)) >= 2)
    return Results(
        shown, roots, classification, (len(phi), len(beta) - 1), long_run,
        mean, halfway, milestones, timing, settling, band, near_zero,
        overshoots, oscillates, selected >= end,
    )
