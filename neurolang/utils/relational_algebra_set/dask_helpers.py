import numpy as np
from collections import namedtuple
from abc import ABC
from dask_sql import Context
from sqlalchemy.dialects import postgresql


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
    def register_function(cls, f_, fname, params):
        if len(params) == 1:
            # We expect only one param so we just pass it to the lambda func
            cls.get_context().register_function(f_, fname, params, np.bool8)
        else:
            # We expect multiple params, so we wrap them in a tuple / named tuple
            named = False
            try:
                named_tuple_type = namedtuple("lambdatuple", [name for (name, _) in params])
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
            cls.get_context().register_function(wrapped_lambda, fname, params, np.bool8)



