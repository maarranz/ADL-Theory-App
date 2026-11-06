import math

from nicegui import ui

def table_rows(table):
    return [
        {key: (None if not math.isfinite(value) else value) for key, value in row.items()}
        for row in table.to_dict('records')
    ]


def render(state):
    result = state.results
    ui.label(f'Multipliers through lag {result.horizon}').classes('text-lg font-bold')
    if result.table['% Total Effect'].isna().all():
        ui.label('Percentage effects are unavailable without a reliably nonzero, convergent long-run effect.')
    columns = [{'name': name, 'label': name, 'field': name, 'align': 'right', 'sortable': True,
                ':format': '(value) => value == null ? "—" : String(value)'}
               for name in result.table.columns]
    ui.table(columns=columns, rows=table_rows(result.table), row_key='Lag',
             pagination=0).props('virtual-scroll').classes('w-full max-h-96')
    with ui.expansion('Reading the numerical results'):
        ui.label('Percentages divide each multiplier by the analytical long-run multiplier and multiply by 100. '
                 'Signed responses can yield negative percentages or totals above 100%.')
