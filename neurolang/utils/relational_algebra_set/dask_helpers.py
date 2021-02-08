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

