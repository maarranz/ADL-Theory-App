# ADL Theory Explorer

ADL-Theory-App is an interactive educational web application for studying the
theoretical properties of **autoregressive distributed lag (ADL/ARDL) models**.
It was developed for **Applied Time Series Econometrics (ATSE)** at
**Universidad Carlos III de Madrid (UC3M)** and can also be used in other
econometrics courses. Students, instructors, and economists can explore
specified models in a browser without writing Python or R. The app explores
model implications and does not estimate models from data.

## Author and affiliation

**Miguel A. Arranz, PhD**

Universidad Carlos III de Madrid (UC3M)

[maarranz@eco.uc3m.es](mailto:maarranz@eco.uc3m.es) ·
[ORCID 0000-0002-5951-2284](https://orcid.org/0000-0002-5951-2284)

## Main features

- **Properties** shows stability, inverse characteristic roots relative to
  the unit circle, and impact and analytical long-run multipliers.
- **Dynamic Effects** plots lag-by-lag multipliers for a one-period unit
  increase in the explanatory variable.
- **Cumulative Effects** plots the response to a sustained unit increase
  and summarizes mean and median lags, adjustment milestones, and settling.
- **Values** gives the numerical multipliers and their shares of the long-run
  effect where defined. The same percentages appear in the plot hover labels.

## Teaching examples and use

Choose an example in the sidebar, then change its coefficients or model
orders to see the equation and results update. The seven examples cover
immediate and delayed responses, persistent adjustment, damped oscillations,
zero net effects, unit roots, and explosive dynamics. The interface supports
orders from 0 to 2 for both the autoregressive and distributed-lag terms.

Use **Auto** for a settling-based display horizon or **Fixed** to choose the
last lag yourself, from 0 to 500. Model settings are independent across
browser clients.

## Installation and local execution

From the checkout, create the Mamba environment if it does not already
exist. The environment uses Python 3.13 and includes the test dependencies.

```sh
mamba env create -f environment.yml
```

Activate the named environment and start the app:

```sh
mamba activate adl_theory_app
python app.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080).

## Econometric interpretation and limits

- The 50%, 90%, 95%, and 99% milestones are the first lags at which the
  cumulative multiplier divided by its analytical long-run value reaches
  those shares. The search runs through lag 500 independently of the
  displayed horizon. A response can overshoot and leave a threshold after
  crossing it.
- Auto checks for settling inside a 1% band and uses a bound on the
  remaining response to verify that it stays there. If the tail cannot be
  verified, the app displays the finite calculation. For a long-run
  magnitude at most 10⁻⁸ times the response scale, the band is absolute,
  at 1% of the largest of 1, the absolute distributed-lag coefficients,
  and the peak absolute cumulative effect through lag 500.
- The mean lag is analytical. A 50% first crossing is reported as a median
  only when the app can establish nonnegative normalized multipliers.
  A signed mean can fall outside the observed lag range.
- For unstable models, Auto displays lags 0–30 and the convergent
  long-run multiplier is unavailable. Percentage shares are also
  unavailable when the long-run multiplier is numerically zero. The
  engine's near-zero denominator check can suppress the multiplier for
  stable models very close to a positive unit root.

## Relationship to ADL-Theory

The app uses the [ADL-Theory Python package](https://github.com/maarranz/adl-theory)
through a vendored copy of `adl_theory.py` from
[revision `7ac6fcd2f98aebf6455c2ccef9bf429a22da055e`](https://github.com/maarranz/adl-theory/blob/7ac6fcd2f98aebf6455c2ccef9bf429a22da055e/adl_theory.py).
The sole local correction supplies an initial value of zero for the
maximum inverse-root modulus when effective AR order is zero, so the
empty root array can be handled. The dynamic recurrence and analytical
formulas are unchanged.

## Testing

With `adl_theory_app` active, run:

```sh
python -m pytest -q
```

## Planned Render deployment

For a [Render Python Web Service](https://render.com/docs/deploy-fastapi),
set [`PYTHON_VERSION`](https://render.com/docs/python-version) to `3.13.18`,
the patch version tested locally. Use `pip install -r requirements.txt` to
build, `python app.py` to start, and `/healthz` for the health check.
The app binds to `0.0.0.0:$PORT` in hosted mode and disables browser
launch and reload. Start with a single process.

## License

[MIT License](LICENSE).
