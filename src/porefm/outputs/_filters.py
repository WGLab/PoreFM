"""Small, shared predicate vocabulary for Arrow and native DuckDB readers."""

import math
from numbers import Integral, Real

import pyarrow as pa
import pyarrow.compute as pc


def identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def literal(value):
    if value is None:
        return 'NULL'
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and math.isfinite(value):
        return repr(float(value))
    raise TypeError('Filter values must be strings, finite numbers, booleans, or None')


def predicates(filters, names):
    """Normalize AND predicates; mapping sequences mean membership, including []."""
    if filters is None:
        return []
    items = ([(key, 'in' if isinstance(value, (list, tuple, set)) else '==', value)
              for key, value in filters.items()] if isinstance(filters, dict) else list(filters))
    result = []
    for name, op, value in items:
        if name not in names:
            raise ValueError(f'Unknown filter column {name!r}; available: {list(names)}')
        if op not in ('=', '==', '!=', '<', '<=', '>', '>=', 'in', 'not in'):
            raise ValueError(f'Unsupported filter operator: {op}')
        if op in ('in', 'not in'):
            if isinstance(value, (str, bytes)):
                raise TypeError('Membership filters require a sequence, not a string')
            value = list(value)
            if any(item is None for item in value):
                raise ValueError('Use == None or != None for null filtering')
            for item in value:
                literal(item)
        else:
            literal(value)
            if value is None and op not in ('=', '==', '!='):
                raise ValueError('Null filters support only equality and inequality')
        result.append((name, op, value))
    return result


def sql_filter(items, schema):
    """Build predicates with Arrow/IEEE comparisons for floating NaNs.

    DuckDB orders NaN above finite values. Arrow's ordered/equality comparisons
    with NaN are false, while inequality is true. Null checks and membership
    keep their existing semantics; infinities remain comparable values. Promote
    floating scalar comparisons to double so DuckDB does not round a threshold
    such as Python's 0.05 to the stored float32 column's precision.
    """
    expressions = []
    for name, op, value in items:
        col = identifier(name)
        if op in ('in', 'not in'):
            expression = (f'{col} {op} ({", ".join(map(literal, value))})' if value
                          else ('false' if op == 'in' else 'true'))
        elif value is None:
            expression = f'{col} IS {"NOT " if op == "!=" else ""}NULL'
        else:
            floating = pa.types.is_floating(schema.field(name).type)
            comparison_col = f'CAST({col} AS DOUBLE)' if floating else col
            expression = f'{comparison_col} {"=" if op == "==" else op} {literal(value)}'
            if floating and op != '!=':
                expression = f'(NOT isnan({col}) AND {expression})'
        expressions.append(expression)
    return ' AND '.join(expressions) or 'true'


def filter_table(table, items):
    functions = {'=': pc.equal, '==': pc.equal, '!=': pc.not_equal, '<': pc.less,
                 '<=': pc.less_equal, '>': pc.greater, '>=': pc.greater_equal}
    for name, op, value in items:
        column = table[name]
        if op in ('in', 'not in'):
            mask = pc.is_in(column, value_set=pa.array(value, type=column.type))
            if op == 'not in':
                mask = pc.invert(mask)
                if value:
                    mask = pc.and_(mask, pc.is_valid(column))
        elif value is None:
            mask = pc.is_valid(column) if op == '!=' else pc.is_null(column)
        else:
            mask = functions[op](column, value)
        table = table.filter(mask)
    return table
