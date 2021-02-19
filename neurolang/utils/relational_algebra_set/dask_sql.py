import logging
import re
import types
import uuid
from collections.abc import Iterable
from typing import Tuple

import dask.dataframe as dd
import numpy as np
from neurolang.type_system import Unknown
from sqlalchemy import (
    and_,
    column,
    func,
    literal,
    literal_column,
    select,
    table,
    text,
)
from sqlalchemy.sql import except_, intersect, table, union

import pandas as pd

from . import abstract as abc
from .dask_helpers import DaskContextFactory, try_to_infer_type_of_operation


LOG = logging.getLogger(__name__)


def _new_name(prefix="table_"):
    return prefix + str(uuid.uuid4()).replace("-", "_")


class DaskRelationalAlgebraBaseSet:
    """
    Base class for RelationalAlgebraSets relying on a Dask-SQL backend.
    This class defines no RA operations but has all the logic of creating /
    iterating / fetching of items in the sets.
    """

    _count = None
    _is_empty = None
    _table_name = None
    _table = None
    _container = None
    _init_columns = None
    dtypes = None

    def __init__(self, iterable=None, columns=None):
        self._init_columns = columns
        if isinstance(iterable, DaskRelationalAlgebraBaseSet):
            if columns is None or columns == iterable.columns:
                self._init_from(iterable)
            else:
                self._init_from_and_rename(iterable, columns)
        elif iterable is not None:
            self._create_insert_table(iterable, columns)

    def _init_from(self, other):
        self._table_name = other._table_name
        self._container = other._container
        self._table = other._table
        self._count = other._count
        self._is_empty = other._is_empty
        self._init_columns = other._init_columns
        self.dtypes = other.dtypes

    def _init_from_and_rename(self, other, columns):
        if other._table is not None:
            query = select(
                *[
                    c.label(str(nc))
                    for c, nc in zip(other.sql_columns, columns)
                ]
            ).select_from(other._table)
            self._table = query.subquery()
        self._is_empty = other._is_empty
        self._count = other._count
        if other.dtypes is not None:
            self.dtypes = other.dtypes.rename(
                {other.dtypes.index[i]: c for i, c in enumerate(columns)}
            )

    def _create_insert_table(self, data, columns=None):
        """
        See https://docs.dask.org/en/latest/best-practices.html
        for best pratices on how to create / manage dask arrays.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=columns)
        elif columns is not None:
            data.columns = list(columns)
        data.columns = data.columns.astype(str)
        # partitions should not be too small, yet fit nicely in memory.
        # The amount of RAM available on the machine should be greater
        # than nb of core x partition size.
        # Here we create partitions of less than 500mb based on the
        # original df size.
        df_size = data.memory_usage(deep=True).sum() / (1 << 20)
        npartitions = 1 + int(df_size) // 500
        LOG.info(
            "Creating dask dataframe with {} partitions"
            " from {:0.2f} Mb pandas df.".format(npartitions, df_size)
        )
        ddf = dd.from_pandas(data, npartitions=npartitions)
        self._set_container(ddf, persist=True)
        if len(data.columns) > 0:
            self.dtypes = data.convert_dtypes().dtypes
        self._count = len(data)
        self._is_empty = self._count == 0

    def _set_container(self, ddf, persist=True):
        self._container = ddf
        if persist:
            # Persist triggers an evaluation of the dask dataframe task-graph.
            # This evaluation is asynchronous (if using an asynchronous scheduler).
            # It will return a new dataframe with a shallow graph.
            # See https://distributed.dask.org/en/latest/memory.html#persisting-collections
            self._container = self._container.persist()
            self._table_name = _new_name()
            DaskContextFactory.get_context().create_table(
                self._table_name, ddf
            )
            self._table = table(
                self._table_name, *[column(c) for c in ddf.columns]
            )

    @property
    def set_row_type(self):
        """
        Return typing info for this set.
        """
        if self.arity > 0:
            types = [self.dtypes[c] for c in self.columns]
            # pandas datatypes are not recognized as types, but they have
            # a .type value which is.
            types = map(
                lambda t: t.type if not isinstance(t, type) else t, types
            )
            return Tuple[tuple(types)]
        return Tuple

    @property
    def arity(self):
        return len(self.columns)

    @property
    def columns(self):
        if self._table is None:
            return [] if self._init_columns is None else self._init_columns
        return self._table.c.keys()

    @property
    def sql_columns(self):
        if self._table is None:
            return {}
        return self._table.c

    @property
    def container(self):
        """
        Accessing the container will evaluate the SQL query representing this set and
        persist the results in Dask.
        """
        if self._container is None:
            if self._table is not None and self.arity > 0:
                q = select(self._table)
                ddf = DaskContextFactory.sql(q)
                self._set_container(ddf, persist=True)
        return self._container

    @classmethod
    def dee(cls):
        output = cls()
        output._count = 1
        return output

    @classmethod
    def dum(cls):
        output = cls()
        output._count = 0
        return output

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls()
        output._init_from(other)
        return output

    def copy(self):
        if self.is_dee():
            return self.dee()
        elif self.is_dum():
            return self.dum()
        return type(self).create_view_from(self)

    def _create_view_from_query(self, query, dtypes, is_empty=None):
        output = type(self)()
        output._table = query.subquery()
        output._container = None
        output.dtypes = dtypes
        output._is_empty = is_empty
        return output

    def is_empty(self):
        if self._is_empty is None:
            if self._count is not None:
                self._is_empty = self._count == 0
            else:
                self._is_empty = self.fetch_one() is None
        return self._is_empty

    def is_dum(self):
        return self.arity == 0 and self.is_empty()

    def is_dee(self):
        return self.arity == 0 and not self.is_empty()

    def __len__(self):
        if self._count is None:
            if self.container is None:
                self._count = 0
            else:
                self._count = len(self.container.drop_duplicates())
        return self._count

    def __contains__(self, element):
        if self.arity == 0:
            return False
        element = self._normalise_element(element)
        query = select(self._table)
        for c, v in element.items():
            query = query.where(self.sql_columns.get(c) == v)
        res = DaskContextFactory.sql(query).head(1)
        return len(res) > 0

    def _normalise_element(self, element):
        if isinstance(element, dict):
            pass
        elif hasattr(element, "__iter__") and not isinstance(element, str):
            element = dict(zip(self.columns, element))
        else:
            element = dict(zip(self.columns, (element,)))
        return element

    def itervalues(self):
        raise NotImplementedError()

    def __iter__(self, named=False):
        if self.is_dee():
            return iter([tuple()])
        if named:
            try:
                return self._fetchall(True).itertuples(
                    name="tuple", index=False
                )
            except ValueError:
                # Invalid column names for namedtuple, just return unnamed tuples
                pass
        return self._fetchall(True).itertuples(name=None, index=False)

    def as_numpy_array(self):
        return self._fetchall(True).to_numpy()

    def as_pandas_dataframe(self):
        return self._fetchall(True)

    def _fetchall(self, drop_duplicates=False):
        if self.container is None:
            if self._count == 1:
                return pd.DataFrame([()])
            else:
                return pd.DataFrame([])
        df = self.container.compute()
        if drop_duplicates:
            df = df.drop_duplicates()
        return df

    def fetch_one(self, named=False):
        if self.container is None:
            if self._count == 1:
                return tuple()
            return None
        if not hasattr(self, '_one_row'):
            name = "tuple" if named else None
            try:
                self._one_row = next(
                    self.container.head(1).itertuples(name=name, index=False)
                )
            except StopIteration:
                self._one_row = None
        return self._one_row

    def __eq__(self, other):
        if isinstance(other, DaskRelationalAlgebraBaseSet):
            if self.is_dee() or other.is_dee():
                res = self.is_dee() and other.is_dee()
            elif self.is_dum() or other.is_dum():
                res = self.is_dum() and other.is_dum()
            elif (
                self._table_name is not None
                and self._table_name == other._table_name
            ):
                res = True
            elif not self._equal_sets_structure(other):
                res = False
            else:
                select_left = select(self._table)
                select_right = select(
                    *[other.sql_columns.get(c) for c in self.columns]
                ).select_from(other._table)
                diff_left = select_left.except_(select_right)
                diff_right = select_right.except_(select_left)
                if len(DaskContextFactory.sql(diff_left).head(1)) > 0:
                    res = False
                elif len(DaskContextFactory.sql(diff_right).head(1)) > 0:
                    res = False
                else:
                    res = True
            return res
        else:
            return super().__eq__(other)

    def _equal_sets_structure(self, other):
        return set(self.columns) == set(other.columns)

    def __repr__(self):
        t = self._table
        return "{}({})".format(type(self), t)

    def __hash__(self):
        if self._table is None:
            return hash((tuple(), None))
        return hash((tuple(self.columns), self.as_numpy_array().tobytes()))


class RelationalAlgebraFrozenSet(
    DaskRelationalAlgebraBaseSet, abc.RelationalAlgebraFrozenSet
):
    def __init__(self, iterable=None, columns=None):
        super().__init__(iterable, columns=columns)

    def selection(self, select_criteria):
        if self._table is None:
            return self.copy()

        query = select(self._table)
        if callable(select_criteria):
            lambda_name = _new_name("lambda")
            params = [(c, self.dtypes[c]) for c in self.columns]
            DaskContextFactory.register_function(
                select_criteria, lambda_name, params, np.bool8
            )
            f_ = getattr(func, lambda_name)
            query = query.where(f_(*self.sql_columns))
        elif isinstance(
            select_criteria, abc.RelationalAlgebraStringExpression
        ):
            # replace == used in python by = used in SQL
            query = query.where(text(re.sub("==", "=", str(select_criteria))))
        else:
            for k, v in select_criteria.items():
                if callable(v):
                    lambda_name = _new_name("lambda")
                    c_ = self.sql_columns.get(str(k))
                    DaskContextFactory.register_function(
                        v,
                        lambda_name,
                        [(str(k), self.dtypes[str(k)])],
                        np.bool8,
                    )
                    f_ = getattr(func, lambda_name)
                    query = query.where(f_(c_))
                elif isinstance(
                    select_criteria, abc.RelationalAlgebraStringExpression
                ):
                    query = query.where(text(re.sub("==", "=", str(v))))
                else:
                    query = query.where(self.sql_columns.get(str(k)) == v)
        return self._create_view_from_query(query, dtypes=self.dtypes)

    def selection_columns(self, select_criteria):
        if self._table is None:
            return self.copy()
        query = select(*self.sql_columns).select_from(self._table)
        for k, v in select_criteria.items():
            query = query.where(
                self.sql_columns.get(str(k)) == self.sql_columns.get(str(v))
            )
        return self._create_view_from_query(query, dtypes=self.dtypes)

    def equijoin(self, other, join_indices=None):
        res = self._dee_dum_product(other)
        if res is not None:
            return res

        # Create an alias on the other table's name if we're joining on
        # the same table.
        ot = other._table
        if other._table_name == self._table_name:
            ot = ot.alias()

        join_cols = list(self.sql_columns) + [
            ot.c.get(str(i)).label(str(i + self.arity))
            for i in range(other.arity)
        ]
        query = select(*join_cols)

        if join_indices is not None and len(join_indices) > 0:
            on_clause = and_(
                *[
                    self.sql_columns.get(str(i)) == ot.c.get(str(j))
                    for i, j in join_indices
                ]
            )
            query = query.select_from(
                self._table.join(ot, on_clause)
            ).distinct()
        dtypes = self.dtypes
        for i in range(other.arity):
            dtypes[str(i + self.arity)] = other.dtypes[str(i)]
        return self._create_view_from_query(query, dtypes)

    def cross_product(self, other):
        return self.equijoin(other)

    def groupby(self, columns):
        raise NotImplementedError()

    def projection(self, *columns, reindex=True):
        if self.is_dum():
            return self.dum()
        elif self.is_dee() or len(columns) == 0:
            return self.dee()

        dtypes = self.dtypes[[str(c) for c in columns]]
        if reindex:
            proj_columns = [
                self.sql_columns.get(str(c)).label(str(i))
                for i, c in enumerate(columns)
            ]
            dtypes.index = [str(i) for i in range(len(columns))]
        else:
            proj_columns = [self.sql_columns.get(str(c)) for c in columns]
        query = select(proj_columns).select_from(self._table).distinct()

        return self._create_view_from_query(
            query, dtypes=dtypes, is_empty=self._is_empty
        )

    def _do_set_operation(self, other, sql_operator):
        if not self._equal_sets_structure(other):
            raise ValueError(
                "Relational algebra set operators can only be used on sets"
                " with same columns."
            )

        ot = other._table.alias()
        query = sql_operator(
            select(self._table),
            select([ot.c.get(c) for c in self.columns]).select_from(ot),
        )
        return self._create_view_from_query(query, dtypes=self.dtypes)

    def __and__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__and__(other)
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        return self._do_set_operation(other, intersect)

    def __or__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__or__(other)
        res = self._dee_dum_sum(other)
        if res is not None:
            return res
        return self._do_set_operation(other, union)

    def __sub__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__sub__(other)
        if self.is_dee():
            if other.is_dee():
                return self.dum()
            return self.dee()
        if self._table is None or other._table is None:
            return self.copy()
        return self._do_set_operation(other, except_)


class NamedRelationalAlgebraFrozenSet(
    RelationalAlgebraFrozenSet, abc.NamedRelationalAlgebraFrozenSet
):
    def __init__(self, columns=None, iterable=None):
        if isinstance(columns, RelationalAlgebraFrozenSet):
            iterable = columns
            columns = columns.columns
        self._check_for_duplicated_columns(columns)
        super().__init__(iterable, columns)

    @staticmethod
    def _check_for_duplicated_columns(columns):
        if columns is not None and len(set(columns)) != len(columns):
            columns = list(columns)
            dup_cols = set(c for c in columns if columns.count(c) > 1)
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"Found the following duplicated columns: {dup_cols}"
            )

    @property
    def columns(self):
        return tuple(super().columns)

    def fetch_one(self):
        return super().fetch_one(named=True)

    def __iter__(self):
        return super().__iter__(named=True)

    def projection(self, *columns):
        return super().projection(*columns, reindex=False)

    def cross_product(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if len(set(self.columns).intersection(set(other.columns))) > 0:
            raise ValueError(
                "Cross product with common columns " "is not valid"
            )

        query = select(self._table, other._table).distinct()
        return self._create_view_from_query(
            query,
            dtypes=self.dtypes.append(other.dtypes),
            is_empty=self._is_empty,
        )

    def naturaljoin(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res

        on = [c for c in self.columns if c in other.columns]
        if len(on) == 0:
            return self.cross_product(other)
        return self._do_join(other, on, isouter=False)

    def left_naturaljoin(self, other):
        """
        Same as naturaljoin with outher=True
        """
        on = [c for c in self.columns if c in other.columns]
        if len(on) == 0:
            return self
        return self._do_join(other, on, isouter=True)

    def _do_join(self, other, on, isouter=False):
        """
        Performs the join on the two sets.

        Parameters
        ----------
        other : NamedRelationalAlgebraFrozenSet
            The other set
        on : Iterable[sqlalchemy.Columns]
            The columns to join on
        isouter : bool, optional
            If True, performs a left outer join, by default False

        Returns
        -------
        NamedRelationalAlgebraFrozenSet
            The joined set
        """
        # Create an alias on the other table's name if we're joining on
        # the same table.
        ot = other._table
        if other._table_name == self._table_name:
            ot = ot.alias()

        on_clause = and_(
            *[self._table.c.get(col) == ot.c.get(col) for col in on]
        )
        other_cols = list(set(other.columns) - set(self.columns))
        select_cols = [self._table] + [ot.c.get(col) for col in other_cols]
        query = (
            select(*select_cols)
            .select_from(self._table.join(ot, on_clause, isouter=isouter))
            .distinct()
        )
        dtypes = self.dtypes.append(other.dtypes[other_cols])
        empty = self._is_empty if isouter else None
        return self._create_view_from_query(
            query, dtypes=dtypes, is_empty=empty
        )

    def equijoin(self, other, join_indices, return_mappings=False):
        raise NotImplementedError()

    def rename_column(self, src, dst):
        if (dst) in self.columns:
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"{dst} is already a column name."
            )
        query = select(
            *[
                c.label(str(dst)) if c.name == src else c
                for c in self.sql_columns
            ]
        ).select_from(self._table)
        dtypes = self.dtypes.rename({str(src): str(dst)})
        return self._create_view_from_query(
            query, dtypes=dtypes, is_empty=self._is_empty
        )

    def rename_columns(self, renames):
        # prevent duplicated destination columns
        self._check_for_duplicated_columns(renames.values())
        if not set(renames).issubset(self.columns):
            # get the missing source columns
            # for a more convenient error message
            not_found_cols = set(c for c in renames if c not in self.columns)
            raise ValueError(
                f"Cannot rename non-existing columns: {not_found_cols}"
            )
        query = select(
            *[
                c.label(str(renames.get(c.name))) if c.name in renames else c
                for c in self.sql_columns
            ]
        ).select_from(self._table)
        dtypes = self.dtypes.rename(renames)
        return self._create_view_from_query(
            query, dtypes=dtypes, is_empty=self._is_empty
        )

    def aggregate(self, group_columns, aggregate_function):
        if isinstance(group_columns, str) or not isinstance(
            group_columns, Iterable
        ):
            group_columns = (group_columns,)
        if len(set(group_columns)) < len(group_columns):
            raise ValueError("Cannot group on repeated columns")

        distinct_sub_query = select(self._table).distinct().subquery()
        agg_cols, agg_types = self._build_aggregate_functions(
            group_columns, aggregate_function, distinct_sub_query
        )
        groupby = [distinct_sub_query.c.get(str(c)) for c in group_columns]

        query = select(*(groupby + agg_cols)).group_by(*groupby)
        dtypes = self.dtypes[list(group_columns)].append(agg_types)
        return self._create_view_from_query(
            query, dtypes=dtypes, is_empty=self._is_empty
        )

    def _build_aggregate_functions(
        self, group_columns, aggregate_function, distinct_view
    ):
        """
        Create the list of aggregated destination columns.
        """
        if isinstance(aggregate_function, dict):
            agg_iter = ((k, k, v) for k, v in aggregate_function.items())
        elif isinstance(aggregate_function, (tuple, list)):
            agg_iter = aggregate_function
        else:
            raise ValueError(
                "Unsupported aggregate_function: {} of type {}".format(
                    aggregate_function, type(aggregate_function)
                )
            )
        un_grouped_cols = [
            c_ for c_ in distinct_view.c if c_.name not in group_columns
        ]
        agg_cols = []
        agg_types = pd.Series(dtype="object")
        for dst, src, f in agg_iter:
            if src in distinct_view.c.keys():
                # call the aggregate function on only one column
                c_ = [distinct_view.c.get(src)]
            else:
                # call the aggregate function on all the non-grouped columns
                c_ = un_grouped_cols
            if isinstance(f, types.BuiltinFunctionType):
                f = f.__name__
                rtype = try_to_infer_type_of_operation(f, self.dtypes)
            if callable(f):
                lambda_name = _new_name("lambda")
                params = [(c.name, self.dtypes[c.name]) for c in c_]
                rtype = try_to_infer_type_of_operation(f, self.dtypes)
                DaskContextFactory.register_aggregation(
                    f, lambda_name, params, rtype
                )
                f_ = getattr(func, lambda_name)
            elif isinstance(f, str):
                f_ = getattr(func, f)
                rtype = try_to_infer_type_of_operation(f, self.dtypes)
            else:
                raise ValueError(
                    f"Aggregate function for {src} needs "
                    "to be callable or a string"
                )
            agg_cols.append(f_(*c_).label(str(dst)))
            agg_types[str(dst)] = rtype
        return agg_cols, agg_types

    def extended_projection(self, eval_expressions):
        if self.is_dee():
            return self._extended_projection_on_dee(eval_expressions)
        elif self._table is None:
            return type(self)(
                columns=list(eval_expressions.keys()), iterable=[]
            )

        proj_columns = []
        dtypes = pd.Series(dtype="object")
        for dst_column, operation in eval_expressions.items():
            if callable(operation):
                lambda_name = _new_name("lambda")
                params = [(c, self.dtypes[c]) for c in self.columns]
                rtype = try_to_infer_type_of_operation(operation, self.dtypes)
                DaskContextFactory.register_function(
                    operation, lambda_name, params, rtype
                )
                f_ = getattr(func, lambda_name)
                proj_columns.append(
                    f_(*self.sql_columns).label(str(dst_column))
                )
                dtypes[str(dst_column)] = rtype
            elif isinstance(operation, abc.RelationalAlgebraStringExpression):
                if str(operation) != str(dst_column):
                    proj_columns.append(
                        literal_column(operation).label(str(dst_column))
                    )
                    rtype = try_to_infer_type_of_operation(
                        operation, self.dtypes
                    )
                    dtypes[str(dst_column)] = rtype
                else:
                    proj_columns.append(self.sql_columns.get(str(operation)))
                    dtypes[str(dst_column)] = self.dtypes[str(dst_column)]
            elif isinstance(operation, abc.RelationalAlgebraColumn):
                proj_columns.append(
                    self.sql_columns.get(str(operation)).label(str(dst_column))
                )
                dtypes[str(dst_column)] = self.dtypes[str(operation)]
            else:
                proj_columns.append(literal(operation).label(str(dst_column)))
                rtype = try_to_infer_type_of_operation(operation, self.dtypes)
                dtypes[str(dst_column)] = rtype

        query = select(proj_columns).select_from(self._table).distinct()
        return self._create_view_from_query(
            query, dtypes=dtypes, is_empty=self._is_empty
        )

    def _extended_projection_on_dee(self, eval_expressions):
        """
        Extended projection when called on Dee to create set with
        constant values.
        """
        return type(self)(
            columns=eval_expressions.keys(),
            iterable=[eval_expressions.values()],
        )

    def to_unnamed(self):
        if self._table is not None:
            query = select(
                *[c.label(str(i)) for i, c in enumerate(self.sql_columns)]
            ).select_from(self._table)
            dtypes = self.dtypes.rename(
                {c: str(i) for i, c in enumerate(self.columns)}
            )
            return RelationalAlgebraFrozenSet()._create_view_from_query(
                query, dtypes=dtypes, is_empty=self._is_empty
            )
        return RelationalAlgebraFrozenSet()

    def projection_to_unnamed(self, *columns):
        unnamed_self = self.to_unnamed()
        named_columns = self.columns
        columns = tuple(named_columns.index(c) for c in columns)
        return unnamed_self.projection(*columns)


class RelationalAlgebraSet(
    RelationalAlgebraFrozenSet, abc.RelationalAlgebraSet
):
    def add(self, value):
        raise NotImplementedError()

    def discard(self, value):
        raise NotImplementedError()

    def __ior__(self, other):
        raise NotImplementedError()

    def __isub__(self, other):
        raise NotImplementedError()
