import ast
import logging
import types
from abc import ABC
from collections import namedtuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from neurolang.type_system import (
    Unknown,
    get_args,
    infer_type_builtins,
    typing_callable_from_annotated_function,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import functions

from dask_sql import Context
from dask_sql.mappings import sql_to_python_type

LOG = logging.getLogger(__name__)


class DaskContextFactory(ABC):

    _context = None

    @classmethod
    def get_context(cls):
        if cls._context == None:
            cls._context = Context()
        return cls._context

    @classmethod
    def sql(cls, query):
        return cls.get_context().sql(
            str(
                query.compile(
                    dialect=postgresql.dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            )
        )

    @classmethod
    def register_function(cls, f_, fname, params, return_type):
        if len(params) == 1:
            # We expect only one param so we just pass it to the lambda func
            cls.get_context().register_function(f_, fname, params, return_type)
        else:
            # We expect multiple params, so we wrap them in a tuple / named tuple
            named = False
            try:
                pnames = [name for (name, _) in params]
                named_tuple_type = namedtuple("lambdatuple", pnames)
                named = True
            except ValueError:
                # Invalid column names, just use a tuple instead.
                # named will be False.
                pass

            def wrapped_lambda(*values):
                if named:
                    return f_(named_tuple_type(*values))
                else:
                    return f_(tuple(values))

            cls.get_context().register_function(
                wrapped_lambda, fname, params, return_type
            )

    @classmethod
    def register_aggregation(cls, f_, fname, params, return_type):
        # Note: aggregation in dask is applied in chunks, first to each partition individually,
        # then again to the results of all the chunk aggregations. So transformative aggregation
        # will not work properly, for instance sum(x) - 1 will result in sum(x) - 2 in the end.
        agg = dd.Aggregation(
            fname, lambda chunk: chunk.agg(f_), lambda total: total.agg(f_)
        )
        cls.get_context().register_aggregation(agg, fname, params, return_type)


_SUPPORTED_PANDAS_TYPES = {
    pd.Int64Dtype(),
    pd.Int32Dtype(),
    pd.Int16Dtype(),
    pd.Int8Dtype(),
    pd.BooleanDtype(),
    pd.StringDtype(),
}


def try_to_infer_type_of_operation(
    operation, column_types, default_type=np.float64
):
    """
    Tries to infer the return type for an operation passed to aggregate
    or extended_projection methods.
    In order to work with dask-sql, the return type should be a pandas
    or numpy type.

    Parameters
    ----------
    operation : Union[Callable, str]
        The operation to infer the type for
    column_types : pd.Series
        The dtypes series mapping the dtype for each column.
        Used if operation references a known column.
    default_type : Type, optional
        The return value if type cannot be infered, by default np.float64

    Returns
    -------
    Type
        An infered return type for the operation.
    """
    try:
        # 1. First we try to guess the return type of the operation
        if isinstance(operation, (types.FunctionType, types.MethodType)):
            # operation is a custom function
            rtype = typing_callable_from_annotated_function(operation)
            rtype = get_args(rtype)[1]
        elif isinstance(operation, types.BuiltinFunctionType):
            # operation is something like 'sum'
            rtype = infer_type_builtins(operation)
            rtype = get_args(rtype)[1]
        else:
            # check if it's one of SQLAlchemy's known functions, like count
            if hasattr(functions, operation):
                rtype = getattr(functions, operation).type
                rtype = sql_to_python_type(rtype().compile())
            elif isinstance(operation, str):
                # otherwise operation is probably a str or
                # RelationalAlgebraStringExpression representing a column
                # literal, like 'col_a + 1', or a constant like '0'.
                # We try to parse the expression to get type of variable or
                # constant.
                rtype = type_of_expression(
                    ast.parse(operation, mode="eval").body, column_types
                )
            else:
                rtype = type(operation)
        # 2. Then we convert it to np.dtype (we allow a few pandas dtypes as well)
        if rtype is Unknown:
            rtype = default_type
        elif rtype not in _SUPPORTED_PANDAS_TYPES:
            rtype = np.dtype(rtype)
    except (ValueError, TypeError, NotImplementedError, SyntaxError):
        LOG.warning(
            f"Unable to infer type of operation {operation}"
            f", assuming default {default_type} type instead."
        )
        rtype = default_type
    return rtype


def type_of_expression(node, column_types):
    if isinstance(node, ast.Num):  # <number>
        return type(node.n)
    elif isinstance(node, ast.Constant):  # Constant
        return type(node.value)
    elif isinstance(node, ast.Name):  
        if node.id in column_types.index: # Col name
            return column_types[node.id]
        else:
            return str
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return type_of_expression(node.left, column_types)
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return type_of_expression(node.operand, column_types)
    else:
        return Unknown

