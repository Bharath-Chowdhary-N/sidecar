"""Read and write ECSV and Parquet catalogs."""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from astropy.table import Table


def _write_parquet(table, path):
    # Write to a temporary name then rename so readers never see a partial file.
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    try:
        table.write(tmp, format="parquet", overwrite=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def catalog_output_paths(path, output_format="parquet"):
    """Return output paths; `both` uses the ECSV path as its primary path."""
    OUTPUT_FORMATS = ("ecsv", "parquet", "both")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}")
    path = Path(path)
    expected = ".parquet" if output_format == "parquet" else ".ecsv"
    if path.suffix.lower() != expected:
        raise ValueError(f"output path {path} must end in {expected} for output format {output_format}")
    return (path, parquet_path(path)) if output_format == "both" else (path,)


def write_catalog(table, path, overwrite=True, output_format="parquet"):
    """Write a Parquet, ECSV, or both-format catalog at the specified path."""
    paths = catalog_output_paths(path, output_format)
    if not overwrite:
        for target in paths:
            if target.exists():
                raise FileExistsError(f"Catalog already exists: {target}")
    if output_format in ("ecsv", "both"):
        table.write(paths[0], overwrite=overwrite, format="ascii.ecsv")
    if output_format in ("parquet", "both"):
        _write_parquet(table, paths[-1])


def read_catalog(path):
    """Read a catalog, either parquet or ECSV

    Parameters
    ----------
    path : str or Path
        The catalog filename.
        Filenames endeding in ".parquet" or ".pq" are read as Parquet files.
        Everything else is read as an ECSV file.
    """
    PARQUET_SUFFIXES = (".parquet", ".pq")
    path = Path(path)
    if path.suffix.lower() in PARQUET_SUFFIXES:
        format = "parquet"
    else:
        format = "ascii.ecsv"

    return Table.read(path, format=format)


def _read_one(args):
    path, cache = args
    return read_catalog(path, cache=cache)


def read_catalogs(paths, n_jobs=1, cache=True):
    """Read many catalogs, in parallel across processes if `n_jobs` > 1 (0 or None: all CPUs)."""
    paths = list(paths)
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    n_jobs = min(n_jobs, len(paths))
    if n_jobs <= 1:
        return [read_catalog(p, cache=cache) for p in paths]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(_read_one, [(p, cache) for p in paths], chunksize=4))
