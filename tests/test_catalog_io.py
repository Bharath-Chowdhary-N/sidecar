import os
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from sidecar.catalog_io import parquet_path, read_catalog, read_catalogs, write_catalog


def _table():
    t = Table({"ra": [1.0, 2.0], "dec": [3.0, np.nan], "flag": [True, False], "n": [1, 2]})
    t.meta["detector"] = 7
    return t


def test_write_catalog_writes_ecsv_and_parquet(tmp_path):
    path = tmp_path / "cat.ecsv"
    write_catalog(_table(), path, output_format="both")
    assert path.exists()
    assert parquet_path(path).exists()


def test_read_catalog_prefers_parquet_and_round_trips(tmp_path):
    path = tmp_path / "cat.ecsv"
    write_catalog(_table(), path, output_format="both")
    from_ecsv = Table.read(path, format="ascii.ecsv")
    got = read_catalog(path)
    assert got.colnames == from_ecsv.colnames
    for col in got.colnames:
        assert np.array_equal(got[col], from_ecsv[col], equal_nan=got[col].dtype.kind == "f")
    assert got.meta["detector"] == 7


def test_read_catalog(tmp_path):
    path = tmp_path / "cat.ecsv"
    _table().write(path, format="ascii.ecsv")
    read_catalog(path)
    assert not parquet_path(path).exists()


def test_read_catalogs_parallel_matches_serial(tmp_path):
    paths = []
    for i in range(4):
        t = _table()
        t["n"] = [i, i + 1]
        p = tmp_path / f"cat{i}.ecsv"
        write_catalog(t, p, output_format="both")
        paths.append(p)
    serial = read_catalogs(paths, n_jobs=1)
    parallel = read_catalogs(paths, n_jobs=2)
    assert [list(t["n"]) for t in serial] == [list(t["n"]) for t in parallel]


@pytest.mark.parametrize("output_format", ["ecsv", "parquet", "both"])
def test_write_catalog_formats_and_overwrite(tmp_path, output_format):
    path = tmp_path / ("cat.parquet" if output_format == "parquet" else "cat.ecsv")
    write_catalog(_table(), path, output_format=output_format)
    assert (tmp_path / "cat.ecsv").exists() == (output_format in ("ecsv", "both"))
    assert (tmp_path / "cat.parquet").exists() == (output_format in ("parquet", "both"))
    assert list(read_catalog(path)["n"]) == [1, 2]
    with pytest.raises(FileExistsError):
        write_catalog(_table(), path, overwrite=False, output_format=output_format)


def test_write_catalog_default_and_invalid_path(tmp_path):
    path = tmp_path / "cat.parquet"
    write_catalog(_table(), path)
    assert path.exists()
    assert not (tmp_path / "cat.ecsv").exists()
    with pytest.raises(ValueError, match="must end in"):
        write_catalog(_table(), tmp_path / "wrong.ecsv")
    with pytest.raises(ValueError, match="output_format"):
        write_catalog(_table(), path, output_format="unknown")


def test_both_overwrite_false_checks_companion_before_writing(tmp_path):
    path = tmp_path / "cat.ecsv"
    _table().write(path.with_suffix(".parquet"), format="parquet")
    with pytest.raises(FileExistsError):
        write_catalog(_table(), path, overwrite=False, output_format="both")
    assert not path.exists()


def test_database_save_reads_parquet(tmp_path, monkeypatch):
    from sidecar import database

    path = tmp_path / "detections.parquet"
    write_catalog(Table({"id": [1], "ra": [10.0], "dec": [5.0], "peak_value": [12.0]}), path)
    calls = []
    monkeypatch.setattr(database.Config, "get", lambda: object())
    monkeypatch.setattr(database, "save_one_dia_object", lambda **kwargs: calls.append(kwargs))
    collection = SimpleNamespace(
        get_image=lambda **kwargs: SimpleNamespace(provenance_id=42, mjd=60000)
    )
    database.save_dia_objects_from_subtraction(
        path, 1, 2, "F062", collection, "test-tag", "test-process", threshold=10
    )
    assert len(calls) == 1
    assert calls[0]["name"] == "42_1"
