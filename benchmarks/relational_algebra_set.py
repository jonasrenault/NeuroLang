import logging
import numpy as np
import pandas as pd
from functools import reduce

from neurolang.utils.relational_algebra_set import (
    pandas,
    dask_sql,
)

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False,
        },
        "neurolang.utils.relational_algebra_set": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "dask_sql.context": {
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
logging.config.dictConfig(LOGGING_CONFIG)


class TimeLeftNaturalJoins:
    params = [
        [10 ** 4, 10 ** 5],
        [10],
        [3],
        [6, 12],
        [0.75],
        [pandas, dask_sql],
    ]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "RAS module to test",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, module):
        dfs = _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r)
        self.sets = [
            module.NamedRelationalAlgebraFrozenSet(df.columns, df)
            for df in dfs
        ]

    def time_ra_left_naturaljoin(
        self, N, ncols, njoin_columns, njoins, distinct_r, module
    ):
        res = reduce(lambda a, b: a.left_naturaljoin(b), self.sets)
        post_process_result(self.sets, res)


class TimeChainedNaturalJoins:
    params = [
        [10 ** 4, 10 ** 5],
        [10],
        [3],
        [6, 12],
        [0.75],
        [pandas, dask_sql],
    ]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "RAS module to test",
    ]

    timeout = 60 * 3

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, module):
        dfs = _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r)
        self.sets = [
            module.NamedRelationalAlgebraFrozenSet(df.columns, df)
            for df in dfs
        ]

    def time_ra_naturaljoin_hard(
        self, N, ncols, njoin_columns, njoins, distinct_r, module
    ):
        res = reduce(lambda a, b: a.naturaljoin(b), self.sets)
        post_process_result(self.sets, res)

    def time_ra_naturaljoin_easy(
        self, N, ncols, njoin_columns, njoins, distinct_r, module
    ):
        res = reduce(lambda a, b: a.naturaljoin(b), self.sets[::-1])
        post_process_result(self.sets, res)


class TimeEquiJoin:
    params = [
        [10 ** 4, 10 ** 5],
        [10],
        [3],
        [6, 12],
        [0.75],
        [pandas, dask_sql],
    ]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "RAS module to test",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, module):
        dfs = _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r)
        for d in dfs:
            d.columns = pd.RangeIndex(ncols)
        self.sets = [module.RelationalAlgebraFrozenSet(df) for df in dfs]

    def time_ra_equijoin(
        self, N, ncols, njoin_columns, njoins, distinct_r, module
    ):
        res = reduce(
            lambda a, b: a.equijoin(b, [(i, i) for i in range(njoin_columns)]),
            self.sets,
        )

        post_process_result(self.sets, res)


def post_process_result(sets, result):
    if isinstance(result, dask_sql.RelationalAlgebraFrozenSet):
        # Fetch one seems slower than _fetchall. Need to investigate.
        result._fetchall()
        # result.fetch_one()


def _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r):
    """
    Generate njoins dataframes of decreasing size. The first dataframe has
    shape N x ncols, while each dataframe after that has shape 
    (N / (i + 1) x ncols).
    The first njoin_columns cols of each dataframe have identical rows to
    perform joins on.
    """
    join_columns = [hex(x) for x in range(njoin_columns)]
    rstate = np.random.RandomState(0)
    keys = pd.DataFrame(
        rstate.randint(0, N * distinct_r, size=(N, njoin_columns)),
        columns=join_columns,
    )
    sets = []
    for i in range(njoins):
        # Take a sample of the default keys.
        skeys = keys.sample(frac=1 / (i + 1), random_state=rstate)
        skeys = pd.DataFrame(
            np.tile(skeys.to_numpy(), (njoins - i, 1)), columns=join_columns,
        )
        # Generate random data for the rest of the set
        cols = [hex(ncols * i + x) for x in range(njoin_columns, ncols)]
        df = pd.concat(
            [
                skeys,
                pd.DataFrame(
                    rstate.randint(
                        0,
                        N * distinct_r,
                        size=(skeys.shape[0], ncols - njoin_columns),
                    ),
                    columns=cols,
                ),
            ],
            axis=1,
        )
        sets.append(df)
    return sets
