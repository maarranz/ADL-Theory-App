"""
adl_theory.py

Teaching toolkit for exploring the dynamics of
Autoregressive Distributed Lag (ADL) models.

Author
------
Miguel A. Arranz

License
-------
MIT

Version
-------
0.1.0
"""

from __future__ import annotations

__version__ = "0.1.0"

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# Stability
# ============================================================

def adl_stability(phi):
    """
    Check the stability of an ADL model.

    Parameters
    ----------
    phi : array-like
        Autoregressive coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - characteristic polynomial
        - roots
        - inverse roots
        - stability indicator

    Notes
    -----
    The characteristic polynomial is

        1 - phi1*z - phi2*z² - ... - phip*z^p

    The model is stable if all roots lie outside
    the unit circle.

    Examples
    --------
    >>> phi = [0.8]
    >>> results = adl_stability(phi)
    >>> results["stable"]
    True
    """

    phi = np.asarray(phi, dtype=float)

    # Polynomial coefficients
    polynomial = np.concatenate((-phi[::-1], [1.0]))

    # Characteristic roots
    roots = np.roots(polynomial)

    # Inverse roots
    inverse_roots = 1 / roots

    # Stability
    stable = np.all(np.abs(roots) > 1)

    return {
            "phi": phi,
            "order": len(phi),
            "roots": roots,
            "inverse_roots": inverse_roots,
            "stable": stable,
            "max_inverse_root": np.max(np.abs(inverse_roots), initial=0.0)
    }


# ============================================================
# Graphics
# ============================================================

import numpy as np
import plotly.graph_objects as go


def plot_inverse_roots(stability,
                       title="Inverse Roots of the Characteristic Polynomial"):
    """
    Plot the inverse roots of the characteristic polynomial.

    Parameters
    ----------
    stability : dict
        Dictionary returned by adl_stability().
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.
    """

    inv_roots = stability["inverse_roots"]
    stable = stability["stable"]
    max_root = stability["max_inverse_root"]
    order = stability["order"]

    # ------------------------------------------------------------
    # Unit circle
    # ------------------------------------------------------------

    theta = np.linspace(0, 2*np.pi, 400)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            name="Unit circle",
            line=dict(color="gray", dash="dash")
        )
    )

    # ------------------------------------------------------------
    # Real and imaginary axes
    # ------------------------------------------------------------

    fig.add_hline(y=0,
                  line_color="black",
                  line_width=1)

    fig.add_vline(x=0,
                  line_color="black",
                  line_width=1)

    # ------------------------------------------------------------
    # Inverse roots
    # ------------------------------------------------------------

    colour = "royalblue" if stable else "firebrick"

    hover = []

    for i, r in enumerate(inv_roots):

        hover.append(
            f"<b>Inverse root {i+1}</b><br>"
            f"Real: {r.real:.4f}<br>"
            f"Imaginary: {r.imag:.4f}<br>"
            f"Modulus: {abs(r):.4f}"
        )

    fig.add_trace(
        go.Scatter(
            x=np.real(inv_roots),
            y=np.imag(inv_roots),
            mode="markers+text",
            text=[f"r{i+1}" for i in range(len(inv_roots))],
            textposition="top center",
            marker=dict(
                size=10,
                color=colour
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
            name="Inverse roots"
        )
    )

    # ------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------

    annotation = (
        f"<b>{'Stable' if stable else 'Unstable'} model</b><br>"
        f"ADL({order})<br>"
        f"Max |inverse root| = {max_root:.3f}"
    )

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=annotation,
        showarrow=False,
        align="left",
        bordercolor="black",
        borderwidth=1,
        bgcolor="white"
    )

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------

    fig.update_layout(
        title=title,
        width=700,
        height=700,
        template="plotly_white",
        xaxis_title="Real",
        yaxis_title="Imaginary",
        legend=dict(
            x=0.02,
            y=0.02
        )
    )

    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        zeroline=False
    )

    fig.update_yaxes(
        zeroline=False
    )

    return fig

# ============================================================
# Dynamic effects
# ============================================================




# See chat for integration notes.
import warnings
import numpy as np
import pandas as pd

def adl_dynamic_effects(
    phi,
    beta,
    horizon=None,
    tolerance=0.99,
    max_horizon=500,
):
    """
    Calculate the dynamic effects of an ADL(p, q) model.

    The model is

        y_t = phi_1 y_{t-1} + ... + phi_p y_{t-p}
              + beta_0 x_t + ... + beta_q x_{t-q} + epsilon_t.

    Dynamic effects are calculated recursively as

        psi_0 = beta_0

        psi_h = beta_h
                + sum(phi_i * psi_{h-i})

    where beta_h = 0 for h > q.

    Parameters
    ----------
    phi : array-like
        Autoregressive coefficients [phi_1, ..., phi_p].

    beta : array-like
        Distributed-lag coefficients [beta_0, ..., beta_q].

    horizon : int or None, default=None
        Final lag for which dynamic effects are calculated.

        When None, the function selects the first lag at which the
        cumulative effect reaches `tolerance` times the theoretical
        long-run effect. The search is limited by `max_horizon`.

    tolerance : float, default=0.99
        Proportion of the theoretical long-run effect used to determine
        the automatic horizon.

    max_horizon : int, default=500
        Maximum horizon used when `horizon` is None.

    Returns
    -------
    dict
        Dictionary containing:

        - ``phi``: autoregressive coefficients;
        - ``beta``: distributed-lag coefficients;
        - ``stability``: output from ``adl_stability()``;
        - ``table``: DataFrame with dynamic and cumulative effects;
        - ``summary``: principal model statistics.
    """
    import warnings

    import numpy as np
    import pandas as pd

    phi = np.asarray(phi, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if phi.ndim != 1:
        raise ValueError("`phi` must be a one-dimensional sequence.")

    if beta.ndim != 1:
        raise ValueError("`beta` must be a one-dimensional sequence.")

    if beta.size == 0:
        raise ValueError("`beta` must contain at least beta_0.")

    if not np.all(np.isfinite(phi)):
        raise ValueError("All values in `phi` must be finite.")

    if not np.all(np.isfinite(beta)):
        raise ValueError("All values in `beta` must be finite.")

    if horizon is not None:
        if not isinstance(horizon, (int, np.integer)):
            raise TypeError("`horizon` must be an integer or None.")

        if horizon < 0:
            raise ValueError("`horizon` must be non-negative.")

    if not 0 < tolerance < 1:
        raise ValueError("`tolerance` must lie strictly between 0 and 1.")

    if not isinstance(max_horizon, (int, np.integer)):
        raise TypeError("`max_horizon` must be an integer.")

    if max_horizon < 0:
        raise ValueError("`max_horizon` must be non-negative.")

    stability = adl_stability(phi)
    stable = stability["stable"]

    if not stable:
        warnings.warn(
            "The ADL model is dynamically unstable. Dynamic effects can "
            "still be calculated for a finite horizon, but the theoretical "
            "long-run effect and lag-distribution summaries are not defined.",
            RuntimeWarning,
            stacklevel=2,
        )

    denominator = 1.0 - np.sum(phi)

    if stable and not np.isclose(denominator, 0.0):
        long_run_effect = np.sum(beta) / denominator
    else:
        long_run_effect = np.nan

    p = len(phi)
    q = len(beta) - 1

    def calculate_effect(psi, h):
        beta_h = beta[h] if h <= q else 0.0

        autoregressive_effect = 0.0

        for i in range(1, min(p, h) + 1):
            autoregressive_effect += phi[i - 1] * psi[h - i]

        return beta_h + autoregressive_effect

    if horizon is None:
        psi = []

        for h in range(max_horizon + 1):
            psi.append(calculate_effect(psi, h))

            cumulative_h = np.sum(psi)

            if (
                stable
                and np.isfinite(long_run_effect)
                and not np.isclose(long_run_effect, 0.0)
            ):
                proportion = cumulative_h / long_run_effect

                if proportion >= tolerance:
                    break

        selected_horizon = len(psi) - 1

        if (
            stable
            and np.isfinite(long_run_effect)
            and not np.isclose(long_run_effect, 0.0)
            and selected_horizon == max_horizon
            and cumulative_h / long_run_effect < tolerance
        ):
            warnings.warn(
                f"The cumulative effect did not reach {tolerance:.1%} of "
                f"the theoretical long-run effect by lag {max_horizon}.",
                RuntimeWarning,
                stacklevel=2,
            )

    else:
        selected_horizon = horizon
        psi = []

        for h in range(selected_horizon + 1):
            psi.append(calculate_effect(psi, h))

    psi = np.asarray(psi, dtype=float)
    lags = np.arange(selected_horizon + 1)
    cumulative_effect = np.cumsum(psi)

    if stable and np.isfinite(long_run_effect):
        if np.isclose(long_run_effect, 0.0):
            percent_total_effect = np.full_like(psi, np.nan)
            percent_cumulative_effect = np.full_like(psi, np.nan)
        else:
            percent_total_effect = 100.0 * psi / long_run_effect
            percent_cumulative_effect = (
                100.0 * cumulative_effect / long_run_effect
            )
    else:
        percent_total_effect = np.full_like(psi, np.nan)
        percent_cumulative_effect = np.full_like(psi, np.nan)

    # Analytical mean lag:
    #
    # sum(j * beta_j) / sum(beta_j)
    # +
    # sum(i * phi_i) / (1 - sum(phi_i))
    if (
        stable
        and not np.isclose(np.sum(beta), 0.0)
        and not np.isclose(denominator, 0.0)
    ):
        beta_lags = np.arange(len(beta))
        phi_lags = np.arange(1, len(phi) + 1)

        mean_lag = (
            np.sum(beta_lags * beta) / np.sum(beta)
            + np.sum(phi_lags * phi) / denominator
        )
    else:
        mean_lag = np.nan

    if (
        stable
        and np.isfinite(long_run_effect)
        and not np.isclose(long_run_effect, 0.0)
    ):
        cumulative_share = cumulative_effect / long_run_effect
        median_candidates = np.flatnonzero(cumulative_share >= 0.5)

        if median_candidates.size > 0:
            median_lag = int(median_candidates[0])
        else:
            median_lag = np.nan
    else:
        median_lag = np.nan

    table = pd.DataFrame(
        {
            "Lag": lags,
            "Dynamic Effect": psi,
            "Cumulative Effect": cumulative_effect,
            "% Total Effect": percent_total_effect,
            "% Cumulative Effect": percent_cumulative_effect,
        }
    )

    summary = {
        "stable": stable,
        "horizon": selected_horizon,
        "long_run_effect": long_run_effect,
        "mean_lag": mean_lag,
        "median_lag": median_lag,
        "tolerance": tolerance if horizon is None else None,
        "automatic_horizon": horizon is None,
    }

    return {
        "phi": phi,
        "beta": beta,
        "stability": stability,
        "table": table,
        "summary": summary,
    }


# ==============================================================
# Dynamic effects graphics
# ==============================================================



import plotly.graph_objects as go


def plot_dynamic_effects(results):
    """
    Plot the dynamic multipliers of an ADL model.

    Parameters
    ----------
    results : dict
        Output returned by ``adl_dynamic_effects()``. The dictionary must
        contain:

        - ``results["table"]``: a pandas DataFrame with columns
          ``"Lag"`` and ``"Dynamic Effect"``.
        - ``results["summary"]``: a dictionary containing the long-run
          effect, mean lag, and median lag.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure containing the dynamic-effects graph.
    """
    if not isinstance(results, dict):
        raise TypeError(
            "`results` must be the dictionary returned by "
            "`adl_dynamic_effects()`."
        )

    if "table" not in results:
        raise KeyError("`results` does not contain a `table` entry.")

    if "summary" not in results:
        raise KeyError("`results` does not contain a `summary` entry.")

    table = results["table"]
    summary = results["summary"]

    required_columns = {"Lag", "Dynamic Effect"}
    missing_columns = required_columns.difference(table.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"The results table is missing the following columns: {missing}."
        )

    lags = table["Lag"]
    effects = table["Dynamic Effect"]

    long_run_effect = summary.get("long_run_effect")
    mean_lag = summary.get("mean_lag")
    median_lag = summary.get("median_lag")
    stable = summary.get("stable")

    def format_number(value, decimals=3):
        if value is None:
            return "Not defined"

        try:
            if not np.isfinite(value):
                return "Not defined"
        except TypeError:
            return str(value)

        return f"{value:.{decimals}f}"

    stability_text = {
        True: "Stable",
        False: "Unstable",
        None: "Not available",
    }.get(stable, str(stable))

    annotation_text = (
        f"<b>Model summary</b><br>"
        f"Stability: {stability_text}<br>"
        f"Long-run effect: {format_number(long_run_effect)}<br>"
        f"Mean lag: {format_number(mean_lag)}<br>"
        f"Median lag: {format_number(median_lag, decimals=0)}"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=lags,
            y=effects,
            name="Dynamic effect",
            customdata = table["% Total Effect"],
            hovertemplate=(
                "<b>Lag %{x}</b><br>"
                "Dynamic effect: %{y:.4f}<br>"
                "Share of LR effect: %{customdata:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_width=1,
        line_color="black",
    )

    fig.add_annotation(
        x=0.99,
        y=0.98,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        align="left",
        xanchor="right",
        yanchor="top",
        bordercolor="gray",
        borderwidth=1,
        borderpad=8,
        bgcolor="white",
    )

    fig.update_layout(
        title={
            "text": "Dynamic Effects",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Lag",
        yaxis_title="Dynamic Effect",
        template="plotly_white",
        showlegend=False,
        hovermode="x",
        width=900,
        height=520,
        margin=dict(l=70, r=40, t=80, b=70),
    )

    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=1,
        showgrid=False,
    )

    fig.update_yaxes(
        zeroline=False,
        gridcolor="lightgray",
    )

    return fig

# ============================================================
# Dynamic effect graphics
# ============================================================


def plot_cumulative_effects(results):
    """
    Plot the cumulative dynamic effects of an ADL model.

    Parameters
    ----------
    results : dict
        Output returned by ``adl_dynamic_effects()``. The dictionary must
        contain:

        - ``results["table"]``: a pandas DataFrame with columns
          ``"Lag"``, ``"Cumulative Effect"``, and
          ``"% Cumulative Effect"``.
        - ``results["summary"]``: a dictionary containing the theoretical
          long-run effect, mean lag, median lag, and selected horizon.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure containing the cumulative-effects graph.
    """
    if not isinstance(results, dict):
        raise TypeError(
            "`results` must be the dictionary returned by "
            "`adl_dynamic_effects()`."
        )

    if "table" not in results:
        raise KeyError("`results` does not contain a `table` entry.")

    if "summary" not in results:
        raise KeyError("`results` does not contain a `summary` entry.")

    table = results["table"]
    summary = results["summary"]

    required_columns = {
        "Lag",
        "Cumulative Effect",
        "% Cumulative Effect",
    }

    missing_columns = required_columns.difference(table.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"The results table is missing the following columns: {missing}."
        )

    lags = table["Lag"]
    cumulative_effects = table["Cumulative Effect"]
    cumulative_percentages = table["% Cumulative Effect"]

    long_run_effect = summary.get("long_run_effect")
    mean_lag = summary.get("mean_lag")
    median_lag = summary.get("median_lag")
    horizon = summary.get("horizon")
    tolerance = summary.get("tolerance")
    stable = summary.get("stable")

    def format_number(value, decimals=3):
        if value is None:
            return "Not defined"

        try:
            if not np.isfinite(value):
                return "Not defined"
        except TypeError:
            return str(value)

        return f"{value:.{decimals}f}"

    stability_text = {
        True: "Stable",
        False: "Unstable",
        None: "Not available",
    }.get(stable, str(stable))

    annotation_text = (
        f"<b>Model summary</b><br>"
        f"Stability: {stability_text}<br>"
        f"LR effect: {format_number(long_run_effect)}<br>"
        f"Mean lag: {format_number(mean_lag)}<br>"
        f"Median lag: {format_number(median_lag, decimals=0)}"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=cumulative_effects,
            mode="lines+markers",
            name="Cumulative effect",
            customdata=cumulative_percentages,
            hovertemplate=(
                "<b>Lag %{x}</b><br>"
                "Cumulative effect: %{y:.4f}<br>"
                "Share of LR effect: %{customdata:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    if long_run_effect is not None and np.isfinite(long_run_effect):
        fig.add_hline(
            y=long_run_effect,
            line_width=1.5,
            line_dash="dash",
            line_color="black",
            annotation_text="Long-run effect",
            annotation_position="top left",
        )

    if median_lag is not None and np.isfinite(median_lag):
        median_lag = int(median_lag)

        median_row = table.loc[table["Lag"] == median_lag]

        if not median_row.empty:
            median_effect = median_row["Cumulative Effect"].iloc[0]

            fig.add_trace(
                go.Scatter(
                    x=[median_lag],
                    y=[median_effect],
                    mode="markers",
                    marker=dict(size=10),
                    name="Median lag",
                    hovertemplate=(
                        f"<b>Median lag: {median_lag}</b><br>"
                        f"Cumulative effect: {median_effect:.4f}"
                        "<extra></extra>"
                    ),
                )
            )

    if (
        summary.get("automatic_horizon")
        and horizon is not None
        and tolerance is not None
    ):
        fig.add_vline(
            x=horizon,
            line_width=1,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"{100 * tolerance:.0f}% horizon",
            annotation_position="top right",
        )

    fig.add_annotation(
        x=0.99,
        y=0.05,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        align="left",
        xanchor="right",
        yanchor="bottom",
        bordercolor="gray",
        borderwidth=1,
        borderpad=8,
        bgcolor="white",
    )

    fig.update_layout(
        title={
            "text": "Cumulative Dynamic Effects",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Lag",
        yaxis_title="Cumulative Effect",
        template="plotly_white",
        hovermode="x",
        width=900,
        height=520,
        margin=dict(l=70, r=40, t=80, b=70),
    )

    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=1,
        showgrid=False,
    )

    fig.update_yaxes(
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        gridcolor="lightgray",
    )

    return fig


# ============================================================
# Summary table
# ============================================================

def adl_summary(results):
    """
    Produce a publication-quality summary table for an ADL model.

    Parameters
    ----------
    results : dict
        Output returned by ``adl_dynamic_effects()``.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled table summarising the model's stability and dynamic
        properties.

    Notes
    -----
    The function returns a pandas Styler object. It does not print or
    display the table directly.
    """
    if not isinstance(results, dict):
        raise TypeError(
            "`results` must be the dictionary returned by "
            "`adl_dynamic_effects()`."
        )

    if "summary" not in results:
        raise KeyError("`results` does not contain a `summary` entry.")

    summary = results["summary"]

    required_keys = {
        "stable",
        "horizon",
        "long_run_effect",
        "mean_lag",
        "median_lag",
        "tolerance",
        "automatic_horizon",
    }

    missing_keys = required_keys.difference(summary)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(
            f"The summary dictionary is missing the following keys: {missing}."
        )

    stable = bool(summary["stable"])

    stability_label = "Stable" if stable else "Unstable"

    automatic_horizon = bool(summary["automatic_horizon"])

    horizon_method = (
        f"Automatic ({100 * float(summary['tolerance']):.0f}% threshold)"
        if automatic_horizon
        else "User specified"
    )

    def finite_value(value, *, integer=False):
        if value is None or not np.isfinite(value):
            return "Not defined"

        return int(value) if integer else float(value)

    summary_table = pd.DataFrame(
        {
            "Statistic": [
                "Stability",
                "Long-run effect",
                "Mean lag",
                "Median lag",
                "Selected horizon",
                "Horizon method",
            ],
            "Value": [
                stability_label,
                finite_value(summary["long_run_effect"]),
                finite_value(summary["mean_lag"]),
                finite_value(summary["median_lag"], integer=True),
                int(summary["horizon"]),
                horizon_method,
            ],
        }
    )

    def format_value(value):
        if isinstance(value, float):
            return f"{value:.4f}"

        return str(value)

    styled_table = (
        summary_table.style
        .hide(axis="index")
        .format(
            {
                "Statistic": "{}",
                "Value": format_value,
            }
        )
        .set_caption("ADL Dynamic Effects Summary")
        .set_properties(
            subset=["Statistic"],
            **{
                "font-weight": "bold",
                "text-align": "left",
            },
        )
        .set_properties(
            subset=["Value"],
            **{
                "text-align": "right",
            },
        )
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("caption-side", "top"),
                        ("font-size", "1.1em"),
                        ("font-weight", "bold"),
                        ("text-align", "left"),
                        ("padding-bottom", "8px"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("font-weight", "bold"),
                        ("text-align", "left"),
                        ("border-bottom", "2px solid black"),
                        ("padding", "6px 12px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border-bottom", "1px solid #dddddd"),
                        ("padding", "6px 12px"),
                    ],
                },
                {
                    "selector": "tbody tr:last-child td",
                    "props": [
                        ("border-bottom", "2px solid black"),
                    ],
                },
            ]
        )
    )

    return styled_table
