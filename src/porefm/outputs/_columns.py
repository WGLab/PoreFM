"""Logical prototype columns retained across old and new file schemas."""


def is_probability(name):
    return name in ('probability', 'probabilities') or name.startswith('probability_')


def without_probabilities(columns):
    return [name for name in columns if not is_probability(name)]
