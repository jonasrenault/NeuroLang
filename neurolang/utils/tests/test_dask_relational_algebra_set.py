from typing import Callable, Tuple

import numpy as np
import pandas as pd
from neurolang.utils.relational_algebra_set import (
    RelationalAlgebraStringExpression,
)
from neurolang.utils.relational_algebra_set.dask_helpers import (
    try_to_infer_type_of_operation,
)
from neurolang.utils.relational_algebra_set.dask_sql import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
)
from unittest.mock import patch
import pytest


def test_set_init():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    ras_a = RelationalAlgebraFrozenSet(a)
    ras_b = RelationalAlgebraFrozenSet(b)
    assert not ras_a.is_empty()
    assert not ras_b.is_empty()


def test_named_set_init():
    assert not NamedRelationalAlgebraFrozenSet.dee().is_empty()
    assert NamedRelationalAlgebraFrozenSet.dum().is_empty()
    assert NamedRelationalAlgebraFrozenSet(iterable=[]).is_empty()
    assert NamedRelationalAlgebraFrozenSet(
        columns=("x",), iterable=[]
    ).is_empty()
    a = [(i, i * 2) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("x", "y"), a)
    assert not ras_a.is_empty()
    assert ras_a.columns == ("x", "y")


def test_set_length():
    assert len(RelationalAlgebraFrozenSet.dee()) == 1
    assert len(RelationalAlgebraFrozenSet.dum()) == 0
    assert len(RelationalAlgebraFrozenSet([])) == 0
    ras_a = RelationalAlgebraFrozenSet([(i, i * 2) for i in range(5)])
    assert len(ras_a) == 5
    assert len(ras_a - ras_a) == 0


def test_fetch_one():
    assert RelationalAlgebraFrozenSet.dee().fetch_one() == tuple()
    assert RelationalAlgebraFrozenSet.dum().fetch_one() is None
    assert RelationalAlgebraFrozenSet([]).fetch_one() is None

    a = [(i, i * 2) for i in range(5)]
    ras_a = RelationalAlgebraFrozenSet(a)
    assert ras_a.fetch_one() in a
    assert (ras_a - ras_a).fetch_one() is None


def test_named_fetch_one():
    assert NamedRelationalAlgebraFrozenSet.dee().fetch_one() == tuple()
    assert NamedRelationalAlgebraFrozenSet.dum().fetch_one() is None
    assert NamedRelationalAlgebraFrozenSet(("x",), []).fetch_one() is None

    a = [(i, i * 2) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("x", "y"), a)
    assert ras_a.fetch_one() in a
    assert (ras_a - ras_a).fetch_one() is None


def test_is_empty():
    assert not RelationalAlgebraFrozenSet.dee().is_empty()
    assert RelationalAlgebraFrozenSet.dum().is_empty()
    assert RelationalAlgebraFrozenSet([]).is_empty()
    ras_a = RelationalAlgebraFrozenSet([(i, i * 2) for i in range(5)])
    assert not ras_a.is_empty()
    assert (ras_a - ras_a).is_empty()


def test_iter():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]
    a += a[3:]
    ras_a = RelationalAlgebraFrozenSet(a)
    res = list(iter(ras_a))
    assert res == a[:6]


def test_named_iter():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]
    a += a[3:]
    ras_a = NamedRelationalAlgebraFrozenSet(("y", "x"), a)
    res = list(iter(ras_a))
    assert res == a[:6]


def test_set_dtypes():
    data = [
        (5, "dog", frozenset({(1, 2), (5, 6)}), np.nan, 45.34, False),
        (10, "cat", frozenset({(5, 6), (8, 9)}), np.nan, np.nan, True),
        (np.nan, "cow", np.nan, np.nan, np.nan, True),
    ]
    ras_a = NamedRelationalAlgebraFrozenSet(
        ("a", "b", "c", "d", "e", "f"), data
    )
    expected_dtypes = [
        pd.Int64Dtype(),
        pd.StringDtype(),
        np.object_,
        pd.Int64Dtype(),
        np.float64,
        pd.BooleanDtype(),
    ]
    assert all(ras_a.dtypes == expected_dtypes)
    ras_b = NamedRelationalAlgebraFrozenSet(
        ("aa", "bb", "cc", "dd", "ee", "ff"), ras_a
    )
    assert all(ras_b.dtypes == expected_dtypes)
    assert all(
        ras_b.dtypes.index.values == ("aa", "bb", "cc", "dd", "ee", "ff")
    )


def test_infer_types():
    data = [
        (5, "dog", frozenset({(1, 2), (5, 6)}), np.nan, 45.34, False),
        (10, "cat", frozenset({(5, 6), (8, 9)}), np.nan, np.nan, True),
        (np.nan, "cow", np.nan, np.nan, np.nan, True),
    ]
    dtypes = (
        pd.DataFrame(data, columns=("a", "b", "c", "d", "e", "f"))
        .convert_dtypes()
        .dtypes
    )

    # lambda expression cannot be infered, should return default type
    assert (
        try_to_infer_type_of_operation(lambda x: x + 1, dtypes) == np.float64
    )
    assert try_to_infer_type_of_operation(
        lambda x: x + 1, dtypes, np.dtype(object)
    ) == np.dtype(object)
    func: Callable[[int], int] = lambda x: x ** 2
    func.__annotations__["return"] = int
    assert try_to_infer_type_of_operation(func, dtypes) == np.int64
    assert try_to_infer_type_of_operation("count", dtypes) == pd.Int32Dtype()
    assert try_to_infer_type_of_operation("sum", dtypes) == np.dtype(object)
    assert (
        try_to_infer_type_of_operation(
            RelationalAlgebraStringExpression("e + 1"), dtypes
        )
        == dtypes["e"]
    )
    assert (
        try_to_infer_type_of_operation(
            RelationalAlgebraStringExpression("a * 12"), dtypes
        )
        == dtypes["a"]
    )
    assert try_to_infer_type_of_operation("0", dtypes) == np.int64
    assert (
        try_to_infer_type_of_operation(1.0, dtypes, np.dtype(object))
        == np.float64
    )
    assert try_to_infer_type_of_operation("hello", dtypes) == np.str_
    assert try_to_infer_type_of_operation("hello world", dtypes) == np.float64


def test_row_type():
    data = [
        (5, "dog", frozenset({(1, 2), (5, 6)}), np.nan, 45.34, False),
        (10, "cat", frozenset({(5, 6), (8, 9)}), np.nan, np.nan, True),
        (np.nan, "cow", np.nan, np.nan, np.nan, True),
    ]
    ras_a = NamedRelationalAlgebraFrozenSet(
        ("a", "b", "c", "d", "e", "f"), data
    )
    expected_dtypes = [
        np.int64,
        str,
        np.object_,
        np.int64,
        np.float64,
        np.bool8,
    ]
    assert ras_a.set_row_type == Tuple[tuple(expected_dtypes)]
    assert NamedRelationalAlgebraFrozenSet.dee().set_row_type == Tuple
    assert (
        NamedRelationalAlgebraFrozenSet(("x", "y"), []).set_row_type
        == Tuple[tuple([np.int64, np.int64])]
    )


def x_test_aggregate():
    initial_set = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 1), (7, 8, 9)]
    )
    expected_lambda = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 8)]
    )

    new_set = initial_set.aggregate(["x", "y"], {"z": lambda x: max(x) - 1})
    print(list(new_set))
    assert expected_lambda == new_set


def test_create_view_from():
    a = [(i, i * 2) for i in range(5)]
    ras_a = RelationalAlgebraFrozenSet(a)
    ras_a = ras_a.selection({0: 1})

    ras_b = RelationalAlgebraFrozenSet.create_view_from(ras_a)
    assert ras_b._container is None
    ras_a.fetch_one()
    assert ras_a._container is not None
    assert ras_b._container is None
    assert ras_b == ras_a