"""Teaching specifications in order from finite responses to unstable models."""

EXAMPLES = {
    'ADL(0,0) · Immediate effect': (
        (), (1.0,),
        'A change in x affects y only in the same period.',
    ),
    'ADL(0,2) · Delayed effect': (
        (), (0.0, 0.4, 0.6),
        'There is no impact at lag zero. The response arrives over the next two periods.',
    ),
    'ADL(1,0) · Persistent adjustment': (
        (0.85,), (1.0,),
        'Positive feedback spreads the effect over many periods, even with no lagged x terms.',
    ),
    'ADL(1,0) · Damped oscillations': (
        (-0.8,), (1.0,),
        'Negative feedback makes the response alternate. It crosses the long-run level well before settling.',
    ),
    'ADL(1,1) · Zero net effect': (
        (0.6,), (1.0, -1.0),
        'The initial increase and later decreases cancel out. The long-run effect is zero.',
    ),
    'ADL(1,0) · Unit root': (
        (1.0,), (1.0,),
        'A one-period change has a lasting effect. With a sustained increase in x, the response keeps accumulating.',
    ),
    'ADL(1,0) · Explosive dynamics': (
        (1.12,), (1.0,),
        'Feedback amplifies the response over time, so there is no finite long-run limit.',
    ),
}
DEFAULT_EXAMPLE = 'ADL(1,0) · Persistent adjustment'
