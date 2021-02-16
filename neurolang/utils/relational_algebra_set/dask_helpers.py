import dask.dataframe as dd
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
    def register_function(cls, f_, fname, params, return_type):
        if len(params) == 1:
            # We expect only one param so we just pass it to the lambda func
            cls.get_context().register_function(f_, fname, params, return_type)
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
            cls.get_context().register_function(wrapped_lambda, fname, params, return_type)


    @classmethod
    def register_aggregation(cls, f_, fname, params, return_type):
        # Note: aggregation in dask is applied in chunks, first to each partition individually,
        # then again to the results of all the chunk aggregations. So transformative aggregation
        # will not work properly, for instance sum(x) - 1 will result in sum(x) - 2 in the end.
        agg= dd.Aggregation(fname, lambda chunk: chunk.agg(f_), lambda total: total.agg(f_))
        cls.get_context().register_aggregation(agg, fname, params, return_type)

