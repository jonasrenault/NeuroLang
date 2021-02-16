from neurolang.utils.relational_algebra_set.dask_sql import (
    RelationalAlgebraFrozenSet,
    NamedRelationalAlgebraFrozenSet,
)


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

def test_aggregate():
    initial_set = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 1), (7, 8, 9)]
    )
    expected_lambda = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 8)]
    )

    new_set = initial_set.aggregate(["x", "y"], {"z": lambda x: max(x) - 1})
    print(list(new_set))
    assert expected_lambda == new_set