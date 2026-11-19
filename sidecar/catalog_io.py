"""Read and write ECSV and Parquet catalogs."""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from astropy.table import Table

# Global to be referenced by other parts to get a consistent list of output formats.
# and parquet suffixes
SUFFIXES = {"ecsv": (".ecsv"), "parquet": (".parquet", ".pq")}
OUTPUT_FORMATS = tuple(SUFFIXES.keys())

def _write_parquet(table, path):
    # Write to a temporary name then rename so readers never see a partial file.
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    try:
        table.write(tmp, format="parquet", overwrite=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_catalog(table, path, overwrite=True, output_format="parquet"):
    """Write a Parquet or ECSV format catalog at the specified path."""
    if not overwrite:
        if path.exists():
            raise FileExistsError(f"Catalog already exists: {path}")
    table.write(path, overwrite=overwrite, format=output_format)


def read_catalog(path):
    """Read a catalog, either parquet or ECSV

    Parameters
    ----------
    path : str or Path
        The catalog filename.
        Filenames endeding in ".parquet" or ".pq" are read as Parquet files.
        Everything else is read as an ECSV file.
    """
    path = Path(path)
    if path.suffix.lower() in SUFFIXES["parquet"]:
        format = "parquet"
    else:
        format = "ascii.ecsv"

    return Table.read(path, format=format)


def read_catalogs(paths, n_jobs=1):
    """Read many catalogs, in parallel across processes if `n_jobs` > 1 (0 or None: all CPUs)."""
    paths = list(paths)
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    n_jobs = min(n_jobs, len(paths))
    if n_jobs <= 1:
        return [read_catalog(p) for p in paths]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(read_catalog, [p for p in paths], chunksize=4))
