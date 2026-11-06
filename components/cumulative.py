from nicegui import ui

from analysis import MAX_HORIZON, MILESTONE_SHARES, TOLERANCE
from components.dynamics import effect_figure
from components.properties import format_value


def _compact_table(title, label, entries):
    ui.label(title).classes('text-lg font-bold')
    columns = [
        {'name': 'label', 'label': label, 'field': 'label', 'align': 'left'},
        {'name': 'value', 'label': 'Result', 'field': 'value', 'align': 'right'},
    ]
    ui.table(columns=columns, rows=[{'label': name, 'value': value} for name, value in entries],
             row_key='label', pagination=0).props('dense flat bordered hide-bottom').classes('w-full')


def _lag(value):
    return 'Not established' if value is None else str(value)


def _adjustment_tables(result):
    signed = result.timing in {'Signed', 'Sign not certified'}
    summary = [
        ('Analytical long-run multiplier', format_value(result.long_run)),
        ('Signed mean lag' if signed else 'Mean lag', format_value(result.mean)),
    ]
    if result.milestones is not None:
        summary.append(('Median lag' if result.timing == 'Distribution' else '50% first crossing',
                        _lag(result.halfway)))
    settling = _lag(result.settling) if result.band is not None else 'Unavailable'
    summary.append((f'Settling horizon ({TOLERANCE:.0%} band)', settling))
    milestones = [(f'{percent}%', 'Unavailable' if result.milestones is None
                   else _lag(result.milestones[percent])) for percent in MILESTONE_SHARES]

    with ui.grid().classes('w-full grid-cols-1 md:grid-cols-2 gap-4'):
        with ui.column().classes('w-full min-w-0'):
            _compact_table('Model summary', 'Statistic', summary)
            if signed:
                reason = ('Normalized multipliers include negative weights.' if result.timing == 'Signed'
                          else 'Nonnegative normalized weights could not be established.')
                ui.label(f'{reason} A 50% crossing alone cannot establish a distributional median.').classes('text-sm')
            if result.long_run is None:
                ui.label('A reliable convergent long-run effect is unavailable for this model.').classes('text-sm')
            elif result.mean is None:
                ui.label('The distributed-lag coefficients sum to zero or nearly cancel, so the mean lag is unavailable.').classes('text-sm')
        with ui.column().classes('w-full min-w-0'):
            _compact_table('Adjustment milestones', 'Share first reached', milestones)
            if result.milestones is None:
                ui.label('Shares require a reliably nonzero, convergent long-run effect.').classes('text-sm')
            else:
                ui.label(f'First crossings use the response through lag {MAX_HORIZON}, '
                         'even when the graph shows fewer lags.').classes('text-sm')



def render(state):
    result = state.results
    label = f'{"Fixed" if state.fixed_horizon is not None else "Auto"} · through lag {result.horizon}'
    if result.classification != 'Stable':
        label += ' · finite display for a nonstable model'
    elif result.settling is None:
        label += ' · settling not verified'
    elif state.fixed_horizon is None:
        label += ' · settling verified'
    if result.truncated:
        label += ' · truncated before numerical overflow'
    ui.label(label).classes('text-lg font-bold')
    behavior = []
    if result.overshoots:
        behavior.append('Overshoots the long-run multiplier')
    if result.oscillates:
        behavior.append('Oscillatory adjustment')
    if behavior:
        ui.label(' · '.join(behavior))
    figure = effect_figure(result.table, 'Cumulative Effect')
    if result.long_run is not None:
        figure.add_hline(y=result.long_run, line_dash='dash', line_color='royalblue',
                         annotation_text='Analytical long run')
    ui.plotly(figure).classes('w-full')
    _adjustment_tables(result)
    with ui.expansion('Reading the cumulative response'):
        ui.label('Each point adds the dynamic multipliers from lag zero through that lag. '
                 'It shows the response to a unit increase in x that stays in place.')
        ui.label('Mean lag is the multiplier-weighted delay. A signed mean can fall outside the observed lags. '
                 'With nonnegative normalized weights, median lag is the first lag that accumulates half the effect.')
        ui.label('Overshooting means the response moves beyond its long-run level in the direction '
                 'of the eventual change. Oscillatory adjustment involves repeated reversals.')
    with ui.expansion('How Auto chooses the horizon'):
        ui.label('The response must stay inside the band, including a bound on all later lags. '
                 'The first crossing records when the response first reaches a share of the long-run effect. '
                 'The response can later leave the band. The calculation checks through lag 500.')
        if result.band is not None:
            kind = 'scale-aware absolute' if result.near_zero else 'relative to the long-run magnitude'
            ui.label(f'Band half-width {result.band:.6g} · 1% {kind}.')
        if result.near_zero:
            ui.label('The scale is the largest of 1, the absolute input coefficients β, '
                     'and the peak absolute cumulative effect through lag 500.')
