import numpy as np
from astropy.table import Table
from sidecar.util import filter_catalog_by_abs_threshold, write_ds9_regions_from_catalog


def test_catalog_filter(tmp_path):
    simple_table = Table({"ra": [1.0, 2, 3], "dec": [1.0, 2, 3], "s": [10.0, -6, np.nan]})
    for ext, fmt in (("ecsv", "ecsv"), ("parquet", "parquet")):
        test_path = tmp_path / f"c.{ext}"
        simple_table.write(test_path, format="ascii.ecsv" if fmt == "ecsv" else "parquet")
        sel, o = filter_catalog_by_abs_threshold(test_path, "s", 5, prefix="sel_")
        assert len(sel) == 2 and o.suffix == "." + ext
        write_ds9_regions_from_catalog(
            o, ds9_region_path=tmp_path / "x.reg", peak_value_column="s", peak_value_threshold=None
        )
        assert (tmp_path / "x.reg").read_text().count("circle") == 2
