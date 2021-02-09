import logging
import numpy as np
import pandas as pd
from functools import reduce
from neurolang.utils.relational_algebra_set import (
    dask_helpers,
    pandas,
    dask_sql,
)
import sqlalchemy

logger = logging.getLogger("dask_sql.context")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(ch)


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
        """
        Generate njoins NamedRelationalAlgebraFrozenSets each with N elements
        & ncols arity. The first njoin_columns columns are identical to
        perform joins on.

        Parameters
        ----------
        N : int
            number of elements per set
        ncols : int
            number of columns per set
        njoin_columns : int
            number of identical columns for all sets
        njoins : int
            number of joins to chain
        distinct_r: int
            ratio of distinct elements in the set
        """
        join_columns = [hex(x) for x in range(njoin_columns)]
        rstate = np.random.RandomState(0)
        keys = pd.DataFrame(
            rstate.randint(0, N * distinct_r, size=(N, njoin_columns)),
            columns=join_columns,
        )
        self.sets = []
        for i in range(njoins):
            cols = [hex(ncols * i + x) for x in range(njoin_columns, ncols)]
            df = pd.concat(
                [
                    keys,
                    pd.DataFrame(
                        rstate.randint(
                            0, N * distinct_r, size=(N, ncols - njoin_columns)
                        ),
                        columns=cols,
                    ),
                ],
                axis=1,
            )
            self.sets.append(
                module.NamedRelationalAlgebraFrozenSet(join_columns + cols, df)
            )

    def time_ra_left_naturaljoin(
        self, N, ncols, njoin_columns, njoins, distinct_r, module
    ):
        res = reduce(lambda a, b: a.left_naturaljoin(b), self.sets)
        post_process_result(self.sets, res)


class TimeChainedNaturalJoins:
    params = [
        [10 ** 3],
        # [10 ** 4, 10 ** 5],
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
        """
        Generate NamedRelationalAlgebraFrozenSets to test performance impact
        of chain order on natural joins. All sets share the same key columns,
        but the first ones have a large portion of the keys repeated
        multiple times, while the last ones have only a small subset of
        distinct keys and are smaller in size.
        Hence, chaining the joins in reverse order of the sets should be
        much more efficient than chaining them in the direct order.

        Parameters
        ----------
        N : int
            number of elements per set
        ncols : int
            number of columns per set
        njoin_columns : int
            number of identical columns for all sets
        njoins : int
            number of joins to chain
        distinct_r: int
            ratio of distinct elements in the sets
        """
        join_columns = [hex(x) for x in range(njoin_columns)]
        rstate = np.random.RandomState(0)
        keys = pd.DataFrame(
            rstate.randint(0, N * distinct_r, size=(N, njoin_columns)),
            columns=join_columns,
        )
        self.sets = []
        for i in range(njoins):
            # Take a sample of the default keys.
            skeys = keys.sample(frac=1 / (i + 1), random_state=rstate)
            skeys = pd.DataFrame(
                np.tile(skeys.to_numpy(), (njoins - i, 1)),
                columns=join_columns,
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
            self.sets.append(
                module.NamedRelationalAlgebraFrozenSet(join_columns + cols, df)
            )

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
        """
        Generate njoins RelationalAlgebraFrozenSets each with N elements
        & ncols arity. The first njoin_columns columns are identical to
        perform joins on.

        Parameters
        ----------
        N : int
            number of elements per set
        ncols : int
            number of columns per set
        njoin_columns : int
            number of identical columns for all sets
        njoins : int
            number of joins to chain
        distinct_r: int
            ratio of distinct elements in the set
        """
        rstate = np.random.RandomState(0)
        df1 = pd.DataFrame(
            rstate.randint(0, N * distinct_r, size=(N, njoin_columns))
        )
        self.sets = []
        for _ in range(njoins):
            df = pd.DataFrame(
                rstate.randint(
                    0, N * distinct_r, size=(N, ncols - njoin_columns)
                ),
            )
            self.sets.append(
                module.RelationalAlgebraFrozenSet(
                    pd.concat([df1, df], axis=1, ignore_index=True)
                )
            )

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
        # Fetch one element from the result set
        result.fetch_one()
        # Fetch the results from the view to execute the join query
        # query = sqlalchemy.select(result._table)
        # qr = dask_helpers.DaskContextFactory.sql(query)
        # qr.compute()
