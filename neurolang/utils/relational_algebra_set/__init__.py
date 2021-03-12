import os
from .abstract import RelationalAlgebraColumnInt, RelationalAlgebraColumnStr, RelationalAlgebraStringExpression
from .config import config

if config["RAS"].get("Backend", "pandas") == "sql":
# if os.getenv("NEURO_RAS_BACKEND", "pandas") == "sql":
    from .sql import (
        NamedRelationalAlgebraFrozenSet,
        RelationalAlgebraFrozenSet,
        RelationalAlgebraSet,
    )
elif config["RAS"].get("Backend", "pandas") == "dask":
# elif os.getenv("NEURO_RAS_BACKEND", "dask") == "dask":
    from .dask_sql import (
        NamedRelationalAlgebraFrozenSet,
        RelationalAlgebraFrozenSet,
        RelationalAlgebraSet,
    )
else:
    from .pandas import (
        NamedRelationalAlgebraFrozenSet,
        RelationalAlgebraFrozenSet,
        RelationalAlgebraSet,
    )


__all__ = [
    "RelationalAlgebraColumnInt",
    "RelationalAlgebraColumnStr",
    "RelationalAlgebraStringExpression",
    "RelationalAlgebraFrozenSet",
    "RelationalAlgebraSet",
    "NamedRelationalAlgebraFrozenSet",
]
