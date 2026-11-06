"""NiceGUI application assembly. Each page instance owns its model."""
import os

from nicegui import app, ui

from components import cumulative, dynamics, properties, values
from components.sidebar import create_sidebar


@app.get('/healthz')
def health():
    return {'status': 'ok'}


@ui.page('/')
def index():
    ui.page_title('ADL Theory Explorer')
    with ui.header().classes('items-center'):
        menu = ui.button(icon='menu').props('flat round').tooltip('Toggle model controls')
        ui.label('ADL Theory Explorer').classes('text-2xl font-bold')
        ui.space()
        ui.label('ATSE · Theoretical ADL models').classes('text-sm')

    with ui.column().classes('w-full max-w-7xl mx-auto p-4'):
        ui.label('Explore how a change in x affects y over time.')
        with ui.card().classes('w-full'):
            equation = ui.html('', sanitize=False).classes('text-xl w-full overflow-x-auto py-1')
        sections = {
            'Properties': properties,
            'Dynamic Effects': dynamics,
            'Cumulative Effects': cumulative,
            'Values': values,
        }
        with ui.tabs(value='Properties').classes('w-full') as tabs:
            for label in sections:
                ui.tab(label)
        content = ui.column().classes('w-full').props('role=tabpanel')

    state = None

    def render():
        # Mount only the visible chart. Plotly cannot resize hidden tab panels.
        content.clear()
        if state is not None:
            with content:
                sections[tabs.value].render(state)

    def publish(snapshot):
        nonlocal state
        state = snapshot
        equation.set_content(state.equation_mathml())
        tabs.set_visibility(True)
        render()

    def invalidate():
        nonlocal state
        state = None
        equation.set_content('Correct the model input in the sidebar to continue.')
        tabs.set_visibility(False)
        render()

    tabs.on_value_change(lambda _: render())
    drawer = create_sidebar(publish, invalidate)
    menu.on_click(drawer.toggle)


if __name__ in {'__main__', '__mp_main__'}:
    hosted = 'PORT' in os.environ or os.environ.get('RENDER') == 'true'
    ui.run(title='ADL Theory Explorer', host='0.0.0.0' if hosted else '127.0.0.1',
           port=int(os.environ.get('PORT', '8080')), reload=not hosted, show=not hosted)
