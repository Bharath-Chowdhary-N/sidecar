import sys
from types import SimpleNamespace

import numpy as np
import astropy.units as u
import pytest
from astropy.table import MaskedColumn, Table
from roman_datamodels import dqflags

from sidecar import pipeline
from sidecar import reject_stars as reject_stars_cli
from sidecar.catalog_io import read_catalog, write_catalog
from sidecar.pipeline import Detection
from sidecar.star_retrieval import read_star_catalog
from sidecar.util import reject_stars, star_rejection_radii


def _tables():
    stars = Table({"ra": [10.0, 20.0, 30.0], "dec": [0.0, 0.0, 0.0], "phot_g_mean_mag": [10.0, 18.0, np.nan]})
    arcsec = 1 / 3600
    # ra offsets in arcsec from each star
    detection = Table(
        {
            "ra": [
                10.0 + 8 * arcsec,
                10.0 + 40 * arcsec,
                20.0 + 3 * arcsec,
                20.0 + 8 * arcsec,
                30.0 + 3 * arcsec,
                50.0,
            ],
            "dec": [0.0] * 6,
            "id": range(6),
        }
    )
    return detection, stars


def test_constant_radius():
    detection, stars = _tables()
    cleaned = reject_stars(detection, stars, 5 * u.arcsec)
    assert list(cleaned["id"]) == [0, 1, 3, 5]
    assert list(reject_stars(detection, stars, 5.0)["id"]) == [0, 1, 3, 5]


def test_magnitude_dependent_radius():
    detection, stars = _tables()
    # Mag 10 star: 5 * 10**(0.2 * 8) = ~199" (capped to 60"); mag 18 star and NaN-mag star: 5"
    cleaned = reject_stars(detection, stars, 5.0, ref_mag=18.0, max_radius=60.0)
    assert list(cleaned["id"]) == [3, 5]


def test_radius_function():
    radii = star_rejection_radii([10.0, 18.0, 20.0, np.nan], 5.0, ref_mag=18.0, max_radius=100.0)
    np.testing.assert_allclose(radii.to_value(u.arcsec), [100.0, 5.0, 5.0, 5.0])


def test_empty():
    detection, stars = _tables()
    assert len(reject_stars(detection, stars[:0], 5.0)) == len(detection)
    assert len(reject_stars(detection[:0], stars, 5.0)) == 0


@pytest.mark.parametrize("detection_format", ["ecsv", "parquet"])
@pytest.mark.parametrize("star_format", ["ecsv", "parquet"])
@pytest.mark.parametrize("output_format", ["ecsv", "parquet"])
def test_reject_stars_cli_catalog_formats(tmp_path, monkeypatch, detection_format, star_format, output_format):
    detection, stars = _tables()
    detection_path = tmp_path / f"detections.{detection_format}"
    star_path = tmp_path / f"stars.{star_format}"
    write_catalog(detection, detection_path, output_format=detection_format)
    write_catalog(stars, star_path, output_format=star_format)
    monkeypatch.setattr(
        sys, "argv", ["reject_stars", str(detection_path), str(star_path), "--catalog-output-format", output_format]
    )
    reject_stars_cli.main()
    ecsv = tmp_path / "cleaned_detections.ecsv"
    parquet = tmp_path / "cleaned_detections.parquet"
    assert ecsv.exists() == (output_format in ("ecsv",))
    assert parquet.exists() == (output_format in ("parquet",))
    output = parquet if output_format == "parquet" else ecsv
    assert list(read_catalog(output)["id"]) == [0, 1, 3, 5]


def test_reject_stars_cli_default_parquet(tmp_path, monkeypatch):
    detection, stars = _tables()
    detection_path = tmp_path / "detections.ecsv"
    star_path = tmp_path / "stars.ecsv"
    write_catalog(detection, detection_path, output_format="ecsv")
    write_catalog(stars, star_path, output_format="ecsv")
    monkeypatch.setattr(sys, "argv", ["reject_stars", str(detection_path), str(star_path)])
    reject_stars_cli.main()
    assert list(read_catalog(tmp_path / "cleaned_detections.parquet")["id"]) == [0, 1, 3, 5]
    assert not (tmp_path / "cleaned_detections.ecsv").exists()


def test_read_star_catalog_text_compatibility(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n")
    assert len(read_star_catalog(empty)) == 0

    text = tmp_path / "stars.txt"
    text.write_text("ra dec phot_g_mean_mag\n10.0 0.0 18.0\n")
    assert len(read_star_catalog(text)) == 1


def test_read_star_catalog_parquet(tmp_path):
    parquet = tmp_path / "stars.PARQUET"
    write_catalog(Table({"ra": [10.0], "dec": [0.0], "phot_g_mean_mag": [18.0]}), parquet)
    assert list(read_star_catalog(parquet)["ra"]) == [10.0]
    assert list(read_catalog(parquet)["ra"]) == [10.0]

    empty_parquet = tmp_path / "empty.parquet"
    write_catalog(Table(names=["ra", "dec"], dtype=[float, float]), empty_parquet)
    assert len(read_star_catalog(empty_parquet)) == 0

    missing = tmp_path / "missing.parquet"
    write_catalog(Table({"ra": [10.0]}), missing)
    with pytest.raises(ValueError, match="missing required columns.*dec"):
        read_star_catalog(missing)
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"not a parquet file")
    with pytest.raises((ValueError, OSError)):
        read_star_catalog(corrupt)


@pytest.mark.parametrize("output_format", ["ecsv", "parquet"])
def test_clean_score_bad_pixels_and_absolute_snr(tmp_path, output_format):
    detections = Table(
        {
            "id": np.arange(8),
            "ra": [10, 20, 30, 40, 50, 60, 70, 80],
            "dec": np.zeros(8),
            "score_flags_npix_bad": [0, 0, 0, 1, -1, 0, 0, 0],
        }
    )
    detections["score_aper_r3_snr"] = MaskedColumn(
        [7, -9, 6.9, 20, 20, np.nan, 20, 20], mask=[False, False, False, False, False, False, True, False]
    )
    stars = Table({"ra": [80.0], "dec": [0.0]})
    path = tmp_path / "score.ecsv"
    out = tmp_path / f"cleaned.{output_format}"
    write_catalog(detections, path, output_format="ecsv")
    cleaned = Detection.reject_stars(
        stars,
        path,
        5.0,
        out,
        bad_pixel_column="score_flags_npix_bad",
        snr_column="score_aper_r3_snr",
        min_abs_snr=7.0,
        output_format=output_format,
    )
    assert list(cleaned["id"]) == [0, 1]
    assert list(read_catalog(out)["id"]) == [0, 1]
    adjusted = Detection.reject_stars(
        stars,
        path,
        5.0,
        out,
        bad_pixel_column="score_flags_npix_bad",
        snr_column="score_aper_r3_snr",
        min_abs_snr=6.0,
        output_format=output_format,
    )
    assert list(adjusted["id"]) == [0, 1, 2]


@pytest.mark.parametrize("output_format", ["ecsv", "parquet"])
def test_clean_simple_bad_pixels_without_snr_cut(tmp_path, output_format):
    detections = Table(
        {
            "id": [1, 2, 3],
            "ra": [10.0, 20.0, 30.0],
            "dec": [0.0, 0.0, 0.0],
            "simple_diff_flags_npix_bad": [0, 1, -1],
            "simple_diff_aper_r3_snr": [0.5, 100.0, 100.0],
        }
    )
    path = tmp_path / "simple.ecsv"
    write_catalog(detections, path, output_format="ecsv")
    stars = Table({"ra": [], "dec": []})
    cleaned = Detection.reject_stars(
        stars,
        path,
        5.0,
        tmp_path / f"cleaned.{output_format}",
        bad_pixel_column="simple_diff_flags_npix_bad",
        output_format=output_format,
    )
    assert list(cleaned["id"]) == [1]
    with pytest.raises(ValueError, match="provided together"):
        Detection.reject_stars(
            stars,
            path,
            5.0,
            tmp_path / f"invalid.{output_format}",
            snr_column="simple_diff_aper_r3_snr",
            output_format=output_format,
        )


@pytest.mark.parametrize("column_collection", [list, np.array])
@pytest.mark.parametrize("output_format", ["ecsv", "parquet"])
def test_clean_rejects_bad_or_unmeasured_pixels_in_any_image(tmp_path, column_collection, output_format):
    detections = Table(
        {
            "id": np.arange(6),
            "ra": np.arange(6, dtype=float) + 10,
            "dec": np.zeros(6),
            "template_flags_npix_bad": [0, 1, 0, 0, -1, 0],
            "science_flags_npix_bad": [0, 0, 1, 0, 0, 0],
            "score_flags_npix_bad": [0, 0, 0, 1, 0, 0],
        }
    )
    detections["science_flags_npix_bad"] = MaskedColumn(
        detections["science_flags_npix_bad"], mask=[False, False, False, False, False, True]
    )
    path = tmp_path / "detections.ecsv"
    out = tmp_path / f"cleaned.{output_format}"
    write_catalog(detections, path, output_format="ecsv")
    stars = Table({"ra": [], "dec": []})
    columns = column_collection(["template_flags_npix_bad", "science_flags_npix_bad", "score_flags_npix_bad"])
    cleaned = Detection.reject_stars(stars, path, 5.0, out, bad_pixel_column=columns, output_format=output_format)
    assert list(cleaned["id"]) == [0]
    assert list(read_catalog(out)["id"]) == [0]
    with pytest.raises(KeyError):
        Detection.reject_stars(
            stars, path, 5.0, out, bad_pixel_column=["template_flags_npix_bad", "missing"], output_format=output_format
        )


@pytest.mark.parametrize("product", ["score", "simple_diff"])
@pytest.mark.parametrize("output_format", ["ecsv", "parquet"])
def test_center_flag_rejection(tmp_path, product, output_format):
    jump = int(dqflags.pixel.JUMP_DET)
    other = int(dqflags.pixel.PERSISTENCE)
    detections = Table(
        {
            "id": np.arange(8),
            "ra": np.arange(8, dtype=float) + 10,
            "dec": np.zeros(8),
            "template_flags_center": [0, jump, 0, 0, -1, 0, other, 0],
            "science_flags_center": [0, 0, jump | other, 0, 0, 0, 0, 0],
            f"{product}_flags_center": [0, 0, 0, jump, 0, 0, 0, 0],
        }
    )
    detections["science_flags_center"] = MaskedColumn(
        detections["science_flags_center"], mask=[False, False, False, False, False, True, False, False]
    )
    path = tmp_path / "detections.ecsv"
    out = tmp_path / f"cleaned.{output_format}"
    write_catalog(detections, path, output_format="ecsv")
    stars = Table({"ra": [], "dec": []})
    columns = ["template_flags_center", "science_flags_center", f"{product}_flags_center"]
    cleaned = Detection.reject_stars(
        stars, path, 5, out, center_flag_columns=columns, center_dq_flags=jump, output_format=output_format
    )
    assert list(cleaned["id"]) == [0, 6, 7]
    assert list(read_catalog(out)["id"]) == [0, 6, 7]
    cleaned = Detection.reject_stars(
        stars, path, 5, out, center_flag_columns=columns, center_dq_flags=jump | other, output_format=output_format
    )
    assert list(cleaned["id"]) == [0, 7]
    cleaned = Detection.reject_stars(
        stars,
        path,
        5,
        out,
        center_flag_columns="template_flags_center",
        center_dq_flags=jump,
        output_format=output_format,
    )
    assert list(cleaned["id"]) == [0, 2, 3, 5, 6, 7]
    assert (
        len(Detection.reject_stars(stars, path, 5, out, center_flag_columns=columns, output_format=output_format)) == 8
    )
    with pytest.raises(ValueError, match="center_flag_columns"):
        Detection.reject_stars(stars, path, 5, out, center_dq_flags=jump)
    with pytest.raises(ValueError, match="center_flag_columns"):
        Detection.reject_stars(stars, path, 5, out, center_flag_columns=[], center_dq_flags=jump)
    with pytest.raises(ValueError, match="mask"):
        Detection.reject_stars(stars, path, 5, out, center_flag_columns=columns, center_dq_flags=-1)


@pytest.mark.parametrize("product", ["simple_diff", "diff", "score"])
@pytest.mark.parametrize("output_format", ["ecsv", "parquet"])
def test_single_pixel_rejection_keeps_inconclusive_and_point_source(tmp_path, product, monkeypatch, output_format):
    n = 10
    table = Table(
        {
            "id": np.arange(n),
            "ra": np.arange(n, dtype=float) + 10,
            "dec": np.zeros(n),
            "peak_value": [1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
            f"{product}_aper_r3_snr": [10, -10, 10, 10, -10, 10, 10, 10, 10, 10],
        }
    )
    for image in ("science", "template"):
        table[f"{image}_aper_r2_flux"] = np.full(n, 9.5)
        table[f"{image}_aper_r3_flux"] = np.full(n, 10.0)
        table[f"{image}_aper_r3_snr"] = np.full(n, 5.0)
        table[f"{image}_peak"] = np.full(n, 7.0)
        for r in (2, 3, 5):
            table[f"{image}_aper_r{r}_npix_valid"] = np.full(n, np.pi * r**2)
    table["science_aper_r2_flux"][2] = 8.3
    table["science_aper_r3_snr"][3] = 4.9
    table["science_aper_r2_flux"][5] = np.nan
    table["science_aper_r3_flux"] = MaskedColumn(
        table["science_aper_r3_flux"], mask=[False, False, False, False, False, False, True, False, False, False]
    )
    table["science_aper_r5_npix_valid"][7] -= 0.2
    table["science_aper_r3_flux"][9] = 0
    path = tmp_path / "input.ecsv"
    out = tmp_path / f"cleaned.{output_format}"
    write_catalog(table, path, output_format="ecsv")
    stars = Table({"ra": [], "dec": []})
    warnings = []
    monkeypatch.setattr(pipeline.SNLogger, "warning", warnings.append)
    cleaned = Detection.reject_stars(stars, path, 5, out, single_pixel_product=product, output_format=output_format)
    assert list(cleaned["id"]) == [2, 3, 4, 5, 6, 7, 9]
    assert list(read_catalog(out)["id"]) == [2, 3, 4, 5, 6, 7, 9]
    assert any("6 candidates with inconclusive" in warning for warning in warnings)
    relaxed = Detection.reject_stars(
        stars,
        path,
        5,
        out,
        single_pixel_product=product,
        single_pixel_options={"min_r2_r3": 0.96},
        output_format=output_format,
    )
    assert len(relaxed) == n
    assert len(Detection.reject_stars(stars, path, 5, out, output_format=output_format)) == n
    with pytest.raises(ValueError, match="min_peak_r3"):
        Detection.reject_stars(
            stars, path, 5, out, single_pixel_product=product, single_pixel_options={"min_peak_r3": np.nan}
        )
    with pytest.raises(KeyError):
        Detection.reject_stars(stars, path, 5, out, single_pixel_product="missing")


def test_center_flag_options_combine(monkeypatch, tmp_path):
    class TestConfig:
        def value(self, key, default=None):
            if key == "system.ou24.tds_base":
                raise ValueError("not configured")
            return default

    monkeypatch.setattr("sidecar.pipeline.Config.get", lambda: TestConfig())
    kwargs = dict(image_collection=None, data_records=[], temp_dir=tmp_path, output_dir=tmp_path)
    assert Detection(**kwargs).center_dq_flags == 0
    other = int(dqflags.pixel.PERSISTENCE)
    assert Detection(**kwargs, center_dq_flags=other, reject_center_jump_det=True).center_dq_flags == (
        other | int(dqflags.pixel.JUMP_DET)
    )
    with pytest.raises(ValueError, match="mask"):
        Detection(**kwargs, center_dq_flags=2**32)
    with pytest.raises(ValueError, match="center flag columns"):
        Detection(**kwargs, reject_center_jump_det=True, center_flag_columns=[])


@pytest.mark.parametrize("mode", ["default", "custom", "empty"])
def test_pipeline_cli_configures_cleaning_columns(monkeypatch, tmp_path, mode):
    class TestConfig:
        def value(self, key, default=None):
            if key == "system.ou24.tds_base":
                raise ValueError("not configured")
            return default

        def augment_argparse(self, parser):
            pass

        def parse_args(self, args):
            pass

    monkeypatch.setattr(pipeline.Config, "get", lambda *args, **kwargs: TestConfig())
    monkeypatch.setattr(pipeline.ImageCollection, "get_collection", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "make_data_records_from_image_path", lambda **kwargs: [1])
    results = []
    monkeypatch.setattr(Detection, "run_subtractions", lambda self: results.append(self))
    args = ["pipeline", "--science-path", "science", "--temp-dir", str(tmp_path), "--output-dir", str(tmp_path)]
    if mode == "custom":
        args += [
            "--bad-pixel-columns",
            "science_flags_npix_bad",
            "flags_npix_bad",
            "--center-flag-columns",
            "template_flags_center",
            "flags_center",
            "--center-dq-flags",
            "0x4",
            "--no-reject-single-pixel",
            "--single-pixel-min-input-snr",
            "8",
            "--single-pixel-min-r2-r3",
            "0.99",
            "--single-pixel-min-peak-r3",
            "0.8",
            "--single-pixel-area-atol",
            "0.2",
            "--catalog-output-format",
            "ecsv",
        ]
    elif mode == "empty":
        args += ["--bad-pixel-columns", "--reject-center-jump-det"]
    monkeypatch.setattr(sys, "argv", args)
    pipeline.main()
    (detection,) = results
    if mode == "custom":
        assert detection.bad_pixel_columns == ["science_flags_npix_bad", "flags_npix_bad"]
        assert detection.center_flag_columns == ["template_flags_center", "flags_center"]
        assert detection.center_dq_flags == 4
        assert detection.output_format == "ecsv"
        assert not detection.reject_single_pixel
        assert detection.single_pixel_options == {
            "min_input_snr": 8,
            "min_r2_r3": 0.99,
            "min_peak_r3": 0.8,
            "area_atol": 0.2,
        }
    elif mode == "empty":
        assert detection.bad_pixel_columns == []
        assert detection.center_dq_flags == int(dqflags.pixel.JUMP_DET)
    else:
        assert detection.bad_pixel_columns == ["template_flags_npix_bad", "science_flags_npix_bad", "flags_npix_bad"]
        assert detection.center_flag_columns == ["template_flags_center", "science_flags_center", "flags_center"]
        assert detection.center_dq_flags == 0
        assert detection.reject_single_pixel
        assert detection.output_format == "parquet"


@pytest.mark.parametrize("output_format", ["ecsv", "parquet"])
def test_pipeline_output_paths_follow_format(monkeypatch, tmp_path, output_format):
    class TestConfig:
        def value(self, key, default=None):
            if key == "system.ou24.tds_base":
                raise ValueError("not configured")
            return default

    monkeypatch.setattr(pipeline.Config, "get", lambda: TestConfig())
    instance = Detection(None, [], temp_dir=tmp_path, output_dir=tmp_path, output_format=output_format)
    science = {"band": "F062", "observation_id": 1, "sca": 1}
    template = {"band": "F062", "observation_id": 2, "sca": 1}
    paths = instance.path_helper(science, template, science_image_path="science", template_image_path="template")
    suffix = ".parquet" if output_format == "parquet" else ".ecsv"
    for key in (
        "star_catalog_path",
        "difference_truth_path",
        "difference_detection_path",
        "cleaned_score_detection_path",
        "transients_to_detection_path",
    ):
        assert paths[key].suffix == suffix
    assert paths["difference_image_path"].suffix == ".fits"


@pytest.mark.parametrize("output_format", ["ecsv", "parquet"])
def test_pipeline_uses_configured_columns_when_cleaning(monkeypatch, tmp_path, output_format):
    class TestConfig:
        def value(self, key, default=None):
            if key == "system.ou24.tds_base":
                raise ValueError("not configured")
            return default

    monkeypatch.setattr(pipeline.Config, "get", lambda *args, **kwargs: TestConfig())
    paths = {
        key: tmp_path / key
        for key in (
            "full_output_dir",
            "science_image_path",
            "template_image_path",
            "difference_image_path",
            "score_image_path",
            "simple_difference_image_path",
            "simple_difference_detection_path",
            "difference_detection_path",
            "score_detection_path",
            "cleaned_simple_difference_detection_path",
            "cleaned_difference_detection_path",
            "cleaned_score_detection_path",
        )
    }
    paths["simple_difference_image_path"].touch()
    suffix = ".parquet" if output_format == "parquet" else ".ecsv"
    for key in paths:
        if "detection_path" in key:
            paths[key] = paths[key].with_suffix(suffix)
    stars = Table({"ra": [], "dec": []})
    monkeypatch.setattr(pipeline.star_retrieval, "read_star_catalog", lambda path: stars)
    monkeypatch.setattr(pipeline.subtraction, "Pipeline", lambda **kwargs: SimpleNamespace(detection_kwargs={}))
    monkeypatch.setattr(Detection, "path_helper", lambda *args, **kwargs: paths)
    monkeypatch.setattr(Detection, "load_product_image", staticmethod(lambda path: object()))
    monkeypatch.setattr(Detection, "load_no_coverage_mask", staticmethod(lambda path: None))
    detections = Table(
        {
            "id": [0, 1, 2],
            "ra": [10.0, 20.0, 30.0],
            "dec": [0.0, 0.0, 0.0],
            "science_flags_npix_bad": [0, 1, 0],
            "template_flags_npix_bad": [0, 0, 1],
            "science_flags_center": [0, 0, 4],
            "template_flags_center": [0, 4, 0],
            "score_flags_npix_bad": [0, 1, 0],
            "simple_diff_flags_npix_bad": [0, 0, 0],
            "score_flags_center": [0, 0, 4],
            "simple_diff_flags_center": [0, 4, 0],
            "score_aper_r3_snr": [10.0, 10.0, 10.0],
            "simple_diff_aper_r3_snr": [10.0, 10.0, 10.0],
            "diff_aper_r3_snr": [10.0, 10.0, 10.0],
            "peak_value": [1.0, 1.0, 1.0],
        }
    )
    for image in ("science", "template"):
        detections[f"{image}_aper_r2_flux"] = [8.0, 8.0, 9.5]
        detections[f"{image}_aper_r3_flux"] = [10.0, 10.0, 10.0]
        detections[f"{image}_aper_r3_snr"] = [10.0, 10.0, 10.0]
        detections[f"{image}_peak"] = [3.0, 3.0, 7.0]
        for r in (2, 3, 5):
            detections[f"{image}_aper_r{r}_npix_valid"] = [np.pi * r**2] * 3
    monkeypatch.setattr(
        pipeline,
        "detect_and_measure_sources",
        lambda **kwargs: pipeline.write_catalog(
            detections, kwargs["catalog_save_path"], output_format=kwargs["output_format"]
        ),
    )
    instance = Detection(
        image_collection=None,
        data_records=[],
        temp_dir=tmp_path,
        output_dir=tmp_path,
        input_star_catalog_path="stars.ecsv",
        save_debug_products=True,
        center_dq_flags=4,
        bad_pixel_columns=["science_flags_npix_bad", "flags_npix_bad"],
        center_flag_columns=["template_flags_center", "flags_center"],
        output_format=output_format,
    )
    instance.run_one_subtraction(
        SimpleNamespace(get_image=lambda path: object()),
        "F062",
        1,
        1,
        "F062",
        2,
        1,
        assume_subtraction_has_been_run=True,
    )
    assert list(read_catalog(paths["cleaned_score_detection_path"])["id"]) == [0]
    assert list(read_catalog(paths["cleaned_simple_difference_detection_path"])["id"]) == [0]
    assert list(read_catalog(paths["cleaned_difference_detection_path"])["id"]) == [0, 1]
    instance.reject_single_pixel = False
    instance.run_one_subtraction(
        SimpleNamespace(get_image=lambda path: object()),
        "F062",
        1,
        1,
        "F062",
        2,
        1,
        assume_subtraction_has_been_run=True,
    )
    assert list(read_catalog(paths["cleaned_simple_difference_detection_path"])["id"]) == [0, 2]
    assert list(read_catalog(paths["cleaned_difference_detection_path"])["id"]) == [0, 1, 2]
