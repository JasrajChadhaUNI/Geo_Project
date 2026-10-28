"""Optional polygon agreement, kept separate from operational predictions.
Missing reference data are explicitly reported as not run, never as zero accuracy.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize
from rasterio import Affine
from shapely.geometry import box
from shapely.ops import unary_union
from analysis import CRS, dump


def polygon_agreement(p, source_path=None, debris_path=None, coverage_path=None):
    """Evaluate source+debris footprints inside a supplied, documented inventory coverage polygon.
    Each input must contain one polygon layer. Export a selected layer first if necessary.
    Corridor buffers overlap: results are separate and must not be summed.
    """
    columns = [
        "corridor",
        "TP",
        "FP",
        "FN",
        "TN",
        "precision",
        "recall",
        "F1",
        "IoU",
        "evaluated_area_km2",
        "coverage_pct",
    ]
    output = p.out / "gns_polygon_results.csv"
    supplied = [x is not None for x in [source_path, debris_path, coverage_path]]
    if any(supplied) and not all(supplied):
        raise ValueError(
            "Supply all three polygon paths (sources, debris, documented coverage), or leave all three as None."
        )
    if not any(supplied):
        result = pd.DataFrame(columns=columns)
        result.to_csv(output, index=False)
        dump(
            p.out / "polygon_validation_status.json",
            dict(
                status="not_run",
                reason="Supply GNS source polygons, debris polygons and documented inventory coverage. Point proximity is not precision or F1.",
            ),
        )
        return result

    def polygons(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Configured reference file missing: {path}")
        if path.suffix.lower() == ".gpkg" and len(gpd.list_layers(path)) != 1:
            raise ValueError(
                f"Select/export one polygon layer before using {path.name}."
            )
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f"Missing reference CRS: {path}")
        frame = frame.to_crs(CRS)
        if (
            frame.empty
            or not frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()
        ):
            raise ValueError(f"Expected a non-empty polygon layer: {path}")
        return frame.geometry.make_valid()

    truth_geometry = unary_union(
        list(polygons(source_path)) + list(polygons(debris_path))
    )
    coverage_geometry = unary_union(polygons(coverage_path))
    rows = []
    for corridor, road in p.road_unions.items():
        family = "R70" if corridor == "Route 70" else "SH1"
        roi = road.buffer(p.settings.corridor_buffer_m).intersection(coverage_geometry)
        counts = dict(TP=0, FP=0, FN=0, TN=0)
        roi_count = valid_count = 0
        for tile in p.tiles[family].values():
            transform = tile["transform"] * Affine.translation(64, 64)
            bounds = box(
                transform.c, transform.f - 2560, transform.c + 2560, transform.f
            )
            if roi.is_empty or not bounds.intersects(roi):
                continue

            def burn(geometry):
                clipped = geometry.intersection(bounds)
                return (
                    rasterize(
                        [(clipped, 1)],
                        out_shape=(256, 256),
                        transform=transform,
                        fill=0,
                        dtype="uint8",
                    ).astype(bool)
                    if not clipped.is_empty
                    else np.zeros((256, 256), bool)
                )

            inside = burn(roi)
            valid = inside & tile["valid"][64:320, 64:320]
            truth = burn(truth_geometry)
            pred = burn(p.unions[family])
            for key, mask in [
                ("TP", truth & pred),
                ("FP", ~truth & pred),
                ("FN", truth & ~pred),
                ("TN", ~truth & ~pred),
            ]:
                counts[key] += int((valid & mask).sum())
            roi_count += int(inside.sum())
            valid_count += int(valid.sum())
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        ratio = lambda x, y: x / y if y else None
        rows.append(
            dict(
                corridor=corridor,
                **counts,
                precision=ratio(tp, tp + fp),
                recall=ratio(tp, tp + fn),
                F1=ratio(2 * tp, 2 * tp + fp + fn),
                IoU=ratio(tp, tp + fp + fn),
                evaluated_area_km2=valid_count / 10000,
                coverage_pct=100 * valid_count / roi_count if roi_count else None,
            )
        )
    result = pd.DataFrame(rows, columns=columns)
    result.to_csv(output, index=False)
    dump(
        p.out / "polygon_validation_status.json",
        dict(
            status="completed",
            reference_files=[str(x) for x in [source_path, debris_path, coverage_path]],
            pixel_size_m=10,
            rasterisation="pixel centre",
            interpretation="Mapped-inventory area agreement, not road-passability accuracy",
            limitations=[
                "Unmapped small events and reference dates can affect agreement.",
                "Corridor buffers overlap; do not sum their counts.",
                "No threshold tuning is performed using these reference data.",
            ],
        ),
    )
    return result
