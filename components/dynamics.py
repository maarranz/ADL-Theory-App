from math import isfinite

from nicegui import ui
import plotly.graph_objects as go


def effect_figure(table, column):
    multiplier = column.replace('Effect', 'multiplier')
    percent_column, percent_label = {
        'Dynamic Effect': ('% Total Effect', 'Share of analytical long-run multiplier'),
        'Cumulative Effect': ('% Cumulative Effect', 'Share of analytical long-run multiplier attained'),
    }[column]
    percentages = [f'{value:.6g}%' if isfinite(value) else 'Unavailable'
                   for value in table[percent_column]]
    trace = go.Scatter(
        x=table['Lag'].tolist(), y=table[column].tolist(),
        mode='lines+markers',
        customdata=percentages,
        hovertemplate=(f'Lag %{{x}}<br>{multiplier} %{{y:.6g}}'
                       f'<br>{percent_label}: %{{customdata}}<extra></extra>'),
    )
    figure = go.Figure(trace)
    figure.add_hline(y=0, line_color='gray', line_width=1)
    figure.update_layout(
        xaxis_title='Lag', yaxis_title=multiplier, template='plotly_white',
        margin=dict(t=30, b=50, l=65, r=30),
    )
    return figure


def render(state):
    ui.label('Response to a one-period unit change in x').classes('text-lg font-bold')
    ui.plotly(effect_figure(state.results.table, 'Dynamic Effect')).classes('w-full')
    with ui.expansion('Dynamic versus cumulative effects'):
        ui.label('The dynamic multiplier gives the response at each lag to a one-period unit increase in x. '
                 'Adding those multipliers gives the response to a unit increase in x that stays in place.')
