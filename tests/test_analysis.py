import numpy as np
import pytest

from adl_theory import adl_dynamic_effects
from analysis import MAX_HORIZON, analyze
from components.dynamics import effect_figure
from components.values import table_rows


def test_matches_engine_and_analytical_mean():
    result = analyze([0.5], [1.0, 0.5], horizon=8)
    engine = adl_dynamic_effects([0.5], [1.0, 0.5], horizon=8)
    np.testing.assert_allclose(result.table.to_numpy(), engine['table'].to_numpy())
    assert result.long_run == 3
    assert result.mean == pytest.approx(4 / 3)
    assert result.timing == 'Distribution'


def test_finite_distributed_lag_and_effective_orders():
    result = analyze([0, 0], [0, -2, 0])
    assert result.effective_orders == (0, 1)
    assert not len(result.inverse_roots)
    assert result.long_run == -2
    assert result.halfway == result.mean == result.settling == 1
    assert list(result.table['Dynamic Effect']) == [0, -2]


def test_crossing_can_precede_settling_by_many_lags():
    result = analyze([-0.8], [1])
    upstream = adl_dynamic_effects([-0.8], [1])
    assert result.crossing_99 == upstream['summary']['horizon'] == 0
    assert result.settling == result.horizon == 20
    assert result.overshoots and result.oscillates
    assert result.timing == 'Signed'
    extended = adl_dynamic_effects([-0.8], [1], horizon=900)['table']['Cumulative Effect']
    assert abs(extended.iloc[19] - result.long_run) > result.band
    assert (abs(extended.iloc[20:] - result.long_run) <= result.band).all()


def test_second_order_tail_and_unverified_cap():
    result = analyze([1.2, -0.7], [1])
    assert result.settling is not None and result.oscillates
    errors = adl_dynamic_effects([1.2, -0.7], [1], horizon=1000)['table']['Cumulative Effect'] - result.long_run
    assert (abs(errors.iloc[result.settling:]) <= result.band).all()
    slow = analyze([0.999], [1])
    assert slow.settling is None and slow.horizon == MAX_HORIZON


@pytest.mark.parametrize('phi,classification', [([1], 'Unit-circle'), ([-1], 'Unit-circle'),
                                               ([0, -1], 'Unit-circle'), ([1.1], 'Explosive')])
def test_unstable_models_have_finite_display(phi, classification):
    result = analyze(phi, [1])
    assert result.classification.startswith(classification)
    assert result.horizon == 30
    assert result.long_run is result.mean is result.settling is None


@pytest.mark.parametrize('beta', [[1, -1], [1, -1 + 1e-10], [0]])
def test_cancellation_and_zero_response(beta):
    result = analyze([0.5], beta)
    assert result.near_zero
    assert result.settling is not None
    assert result.mean is result.halfway is None
    assert result.table['% Total Effect'].isna().all()
    if beta == [0]:
        assert result.horizon == 0


def test_fixed_zero_and_shared_plot_table_data():
    result = analyze([0.85], [1], horizon=0)
    assert result.horizon == 0 and result.halfway > 0
    for column in ('Dynamic Effect', 'Cumulative Effect'):
        trace = effect_figure(result.table, column).data[0]
        rows = table_rows(result.table)
        assert list(trace.x) == [row['Lag'] for row in rows] == [0]
        assert list(trace.y) == [row[column] for row in rows] == [1]


@pytest.mark.parametrize('phi,beta,horizon', [([np.nan], [1], None), ([], [np.inf], None),
                                             ([], [], None), ([], [1], -1), ([], [1], 0.5)])
def test_invalid_input(phi, beta, horizon):
    with pytest.raises(ValueError):
        analyze(phi, beta, horizon)


@pytest.mark.parametrize('phi,beta', [([0.85], [1]), ([-0.8], [1]), ([], [0, -2]),
                                     ([], [1e9, -1e9 + 1]), ([0.5], [1, -1])])
def test_summary_and_full_precision_columns_match_engine(phi, beta):
    engine = adl_dynamic_effects(phi, beta, horizon=MAX_HORIZON)
    result = analyze(phi, beta, horizon=MAX_HORIZON)
    assert list(result.table.columns) == [
        'Lag', 'Dynamic Effect', 'Cumulative Effect', '% Total Effect', '% Cumulative Effect',
    ]
    np.testing.assert_array_equal(result.table.to_numpy(), engine['table'].to_numpy())
    summary = engine['summary']
    for actual, key in [(result.long_run, 'long_run_effect'),
                        (result.mean, 'mean_lag'), (result.halfway, 'median_lag')]:
        expected = summary[key]
        assert actual == (expected if np.isfinite(expected) else None)
    for column in ('Dynamic Effect', 'Cumulative Effect'):
        figure = effect_figure(result.table, column)
        assert figure.data[0].mode == 'lines+markers'
        assert figure.layout.xaxis.title.text == 'Lag'


def test_small_nonzero_effect_can_have_an_analytical_mean_without_percentage_shares():
    result = analyze([-0.9], [1.1e-8])
    assert result.mean == pytest.approx(-0.9 / 1.9)
    assert result.timing == 'Signed'
    assert result.halfway is None
    assert result.table['% Total Effect'].isna().all()


@pytest.mark.parametrize('phi,beta,expected', [
    ([0.5], [1], [0, 3, 4, 6]),
    ([-0.8], [1], [0, 0, 0, 0]),
    ([0.5], [-1], [0, 3, 4, 6]),
    ([0.999], [1], [None, None, None, None]),
])
def test_milestones_match_cumulative_response_even_with_fixed_zero(phi, beta, expected):
    result = analyze(phi, beta, horizon=0)
    assert list(result.milestones.values()) == expected
    full = adl_dynamic_effects(phi, beta, horizon=MAX_HORIZON)['table']
    for percent, lag in result.milestones.items():
        qualifying = full.loc[full['Cumulative Effect'] / result.long_run >= percent / 100, 'Lag']
        assert lag == (None if qualifying.empty else qualifying.iloc[0])


@pytest.mark.parametrize('phi,beta', [([0.5], [1, -1]), ([1], [1]), ([1.1], [1])])
def test_milestones_unavailable_without_nonzero_convergent_limit(phi, beta):
    assert analyze(phi, beta).milestones is None


def test_mathml_equation_uses_subscripts_and_omits_zero_terms():
    from xml.etree import ElementTree
    from model_state import ModelState

    for phi, beta, terms in [
        ((), (0,), 0), ((), (2,), 1), ((0.5,), (0, -0.25), 2),
        ((-0.3, 0.2), (1, 0, 0.25), 4),
    ]:
        math = ElementTree.fromstring(ModelState.build(phi, beta, None).equation_mathml())
        ns = {'m': 'http://www.w3.org/1998/Math/MathML'}
        assert len(math.findall('.//m:munder', ns)) == terms
        assert math.find('.//m:msub', ns) is not None
        if any(value < 0 for value in (*phi, *beta)):
            assert any(node.text == '−' for node in math.findall('.//m:mo', ns))


@pytest.mark.parametrize('column,percent_column', [
    ('Dynamic Effect', '% Total Effect'),
    ('Cumulative Effect', '% Cumulative Effect'),
])
@pytest.mark.parametrize('phi,beta', [([-0.8], [1]), ([0.5], [1, -1]), ([1], [1])])
def test_hover_shares_use_table_at_every_plotted_lag(column, percent_column, phi, beta):
    table = analyze(phi, beta, horizon=8).table
    trace = effect_figure(table, column).data[0]
    assert tuple(trace.x) == tuple(range(9))
    assert tuple(trace.y) == tuple(table[column])
    assert tuple(trace.customdata) == tuple(
        f'{value:.6g}%' if np.isfinite(value) else 'Unavailable' for value in table[percent_column]
    )
    assert all(token in trace.hovertemplate for token in ('%{x}', '%{y:.6g}', '%{customdata}'))


@pytest.mark.parametrize('phi,beta', [
    ((), (1, -0.25, 0.5)),
    ((0.5,), (1, -0.25)),
    ((-0.5,), (1, -0.25, 0)),
    ((0.5, 0), (-1, 0, 0.5)),
    ((0.5, -0.2), (1, -0.25, 0.5)),
])
def test_mathml_associates_signed_values_with_plain_greek_labels(phi, beta):
    from xml.etree import ElementTree
    from model_state import ModelState

    math = ElementTree.fromstring(ModelState.build(phi, beta, None).equation_mathml())
    ns = {'m': 'http://www.w3.org/1998/Math/MathML'}
    terms = math.findall('./m:mrow/m:munder', ns)
    expected = [(symbol, lag, value)
                for symbol, values, start in [('φ', phi, 1), ('β', beta, 0)]
                for lag, value in enumerate(values, start) if value]
    assert len(terms) == len(expected)
    assert all(node.text != '|' for node in math.findall('.//m:mo', ns))
    for term, (symbol, lag, value) in zip(terms, expected):
        number, label = term
        assert label.tag.endswith('}msub')
        assert [child.text for child in label] == [symbol, str(lag)]
        assert ''.join(number.itertext()) == f'{value:.6g}'.replace('-', '−')
    assert sum(node.text == "+" for node in math.findall("./m:mrow/m:mo", ns)) == (
        1 + sum(value > 0 for _, _, value in expected[1:])
    )
