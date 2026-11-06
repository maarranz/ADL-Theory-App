from nicegui import ui

from adl_theory import plot_inverse_roots


def format_value(value):
    return 'Unavailable' if value is None else f'{value:.6g}'


def render(state):
    result = state.results
    p, q = result.effective_orders
    ui.label(f'{result.classification} · effective ADL({p}, {q})').classes('text-lg font-bold')
    with ui.row().classes('gap-8'):
        ui.label(f'Impact multiplier  {format_value(state.beta[0])}')
        ui.label(f'Analytical long-run multiplier  {format_value(result.long_run)}')
    if result.long_run is None:
        if result.classification == 'Stable':
            ui.label('The long-run denominator is too close to zero for a reliable result.')
        else:
            ui.label('A convergent long-run multiplier is unavailable because the model is not stable.')
    with ui.expansion('Interpreting multipliers'):
        ui.label('The impact multiplier is the immediate response to a unit increase in x. '
                 'The long-run multiplier is the eventual response when that increase stays in place.')
    figure = plot_inverse_roots({
        'inverse_roots': result.inverse_roots,
        'order': p,
        'stable': result.classification == 'Stable',
        'max_inverse_root': max(map(abs, result.inverse_roots), default=0.0),
    }, title='Inverse roots and the unit circle')
    ui.plotly(figure).classes('w-full')
    with ui.expansion('Root details'):
        if not p:
            ui.label('The effective autoregressive order is zero. There are no inverse roots.')
        for index, root in enumerate(result.inverse_roots, 1):
            ui.label(f'{index}. Inverse root {root:.6g} · modulus {abs(root):.6g}')
        ui.label('Inverse roots describe how feedback persists. Inside the unit circle, feedback dies out. '
                 'On the circle it can persist, and outside it can grow.')
