"""Client-local controls with atomic example changes."""
import math

from nicegui import ui

from analysis import MAX_HORIZON
from examples import DEFAULT_EXAMPLE, EXAMPLES
from model_state import ModelState

CUSTOM = 'Custom model'


def create_sidebar(publish, invalidate):
    loading = False
    with ui.left_drawer(value=True).classes('p-4') as drawer:
        ui.label('Model specification').classes('text-xl font-bold')
        example = ui.select([*EXAMPLES, CUSTOM], value=DEFAULT_EXAMPLE,
                            label='Teaching example').classes('w-full')
        explanation = ui.label().classes('text-sm text-gray-600')
        with ui.row().classes('w-full'):
            p = ui.select([0, 1, 2], value=1, label='AR order p').classes('grow')
            q = ui.select([0, 1, 2], value=0, label='Lag order q').classes('grow')
        ui.label('Autoregressive coefficients').classes('font-medium')
        phi = [ui.input(label, value='0').props('inputmode=decimal').classes('w-full')
               for label in ('φ₁', 'φ₂')]
        ui.label('Distributed-lag coefficients').classes('font-medium')
        beta = [ui.input(label, value='0').props('inputmode=decimal').classes('w-full')
                for label in ('β₀', 'β₁', 'β₂')]
        mode = ui.toggle(['Auto', 'Fixed'], value='Auto')
        fixed = ui.input('Fixed horizon', value='30').props('inputmode=numeric').classes('w-full')
        error = ui.label().classes('text-negative text-sm')

    def coefficient(control):
        try:
            value = float(control.value)
        except (ValueError, TypeError):
            raise ValueError(f'Enter a finite number for {control.props["label"]}.') from None
        if not math.isfinite(value):
            raise ValueError(f'Enter a finite number for {control.props["label"]}.')
        return value

    def refresh():
        for index, control in enumerate(phi):
            control.set_visibility(index < p.value)
        for index, control in enumerate(beta):
            control.set_visibility(index <= q.value)
        fixed.set_visibility(mode.value == 'Fixed')
        try:
            horizon = None
            if mode.value == 'Fixed':
                try:
                    horizon = int(fixed.value)
                except (TypeError, ValueError):
                    raise ValueError(f'Enter a whole-number horizon from 0 to {MAX_HORIZON}.') from None
            state = ModelState.build(tuple(map(coefficient, phi[:p.value])),
                                     tuple(map(coefficient, beta[:q.value + 1])), horizon)
        except (ValueError, ArithmeticError) as exc:
            error.set_text(str(exc))
            invalidate()
            return
        error.set_text('')
        publish(state)

    def manual(_):
        if not loading:
            example.set_value(CUSTOM)
            refresh()

    def load(name):
        nonlocal loading
        if name not in EXAMPLES:
            explanation.set_visibility(False)
            return
        autoregressive, distributed, description = EXAMPLES[name]
        explanation.set_text(description)
        explanation.set_visibility(True)
        loading = True
        try:
            p.set_value(len(autoregressive))
            q.set_value(len(distributed) - 1)
            for controls, coefficients in ((phi, autoregressive), (beta, distributed)):
                for index, control in enumerate(controls):
                    control.set_value(str(coefficients[index]) if index < len(coefficients) else '0')
        finally:
            loading = False
        refresh()

    example.on_value_change(lambda event: load(event.value))
    for control in [p, q, *phi, *beta]:
        control.on_value_change(manual)
    mode.on_value_change(lambda _: refresh())
    fixed.on_value_change(lambda _: refresh())
    load(DEFAULT_EXAMPLE)
    return drawer
