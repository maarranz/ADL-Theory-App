"""A validated snapshot and its single analysis result for one client."""
from dataclasses import dataclass

from analysis import Results, analyze


@dataclass(frozen=True)
class ModelState:
    phi: tuple[float, ...]
    beta: tuple[float, ...]
    fixed_horizon: int | None
    results: Results

    @classmethod
    def build(cls, phi, beta, fixed_horizon):
        return cls(phi, beta, fixed_horizon, analyze(phi, beta, fixed_horizon))

    def equation_mathml(self):
        """Native MathML with coefficient labels beneath their numeric values."""
        terms = []
        for coefficients, symbol, variable, start in (
            (self.phi, 'φ', 'y', 1), (self.beta, 'β', 'x', 0),
        ):
            for lag, coefficient in enumerate(coefficients, start):
                if coefficient == 0:
                    continue
                if terms:
                    terms.append('<mspace width="mediummathspace"/>' if coefficient < 0 else '<mo>+</mo>')
                index = f'<mrow><mi>t</mi><mo>−</mo><mn>{lag}</mn></mrow>' if lag else '<mi>t</mi>'
                number = f'{abs(coefficient):.6g}'
                if 'e' in number:
                    mantissa, exponent = number.split('e')
                    magnitude = (f'<mrow><mn>{mantissa}</mn><mo>×</mo>'
                                 f'<msup><mn>10</mn><mn>{int(exponent)}</mn></msup></mrow>')
                else:
                    magnitude = f'<mn>{number}</mn>'
                coefficient_label = f'<msub><mi>{symbol}</mi><mn>{lag}</mn></msub>'
                if coefficient < 0:
                    magnitude = f'<mrow><mo>−</mo>{magnitude}</mrow>'
                terms.append(f'<munder>{magnitude}{coefficient_label}</munder>'
                             f'<mo>⁢</mo><msub><mi>{variable}</mi>{index}</msub>')
        if terms:
            terms.append('<mo>+</mo>')
        return (
            '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'
            '<mrow><msub><mi>y</mi><mi>t</mi></msub><mo>=</mo>'
            + ''.join(terms) + '<msub><mi>ε</mi><mi>t</mi></msub></mrow></math>'
        )
