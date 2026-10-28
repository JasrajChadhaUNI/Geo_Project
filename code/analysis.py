"""Kaikoura retrospective road-screening analysis. No online services required.
GNS data are used only for evaluation, never to alter operational predictions.
"""

from pathlib import Path
import json, math, hashlib, sys, time
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes, rasterize
from shapely.geometry import shape, box, Point, GeometryCollection
from shapely.ops import unary_union
import shapely
from pyproj import Transformer

CRS = "EPSG:2193"
WGS = Transformer.from_crs(CRS, 4326, always_xy=True)
CLASSES = {
    "potential_impact": ("Potential impact", "#c44732"),
    "nearby_detection": ("Nearby detection", "#c58a21"),
    "no_detected_impact": ("No detected impact", "#28786e"),
    "insufficient_observation": ("Insufficient observation", "#75808c"),
}
FOLDERS = {"SH1": "kaikoura_full_coast_SH1", "R70": "kaikoura_route70"}


@dataclass
class Settings:
    threshold: float = 0.5
    minimum_area_m2: float = 300.0
    impact_distance_m: float = 20.0
    nearby_distance_m: float = 100.0
    minimum_visibility: float = 0.8
    corridor_buffer_m: float = 1000.0
    divider_lat: float = -42.40
    seed: int = 761

    def validate(self):
        assert 0 < self.threshold < 1
        assert 0 <= self.minimum_area_m2
        assert 0 <= self.impact_distance_m < self.nearby_distance_m
        assert 0 < self.minimum_visibility <= 1
        assert self.corridor_buffer_m > 0


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8"
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def polygon_parts(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [p for g in getattr(geom, "geoms", []) for p in polygon_parts(g)]


class Project:
    def __init__(self, root, settings=None):
        self.root = Path(root).resolve()
        self.settings = settings or Settings()
        self.settings.validate()
        self.original = self.root / "data/original"
        self.out = self.root / "results"
        self.out.mkdir(exist_ok=True)
        self.tiles = {}
        self.predictions = {}
        self.unions = {}
        self.raw_unions = {}
        self.roads = None

    def load(self):
        """Validate cached raster grids/masks and load the roads and GNS reference points."""
        roads = []
        tile_rows = []
        for family, folder in FOLDERS.items():
            base = self.original / "outputs" / folder
            old = gpd.read_file(base / "04_SH1_road_assessment.geojson").to_crs(CRS)
            old["family"] = family
            old["original_class"] = old["class"]
            old["original_segment_id"] = old["segment_id"]
            lon, lat = WGS.transform(
                old.geometry.representative_point().x.values,
                old.geometry.representative_point().y.values,
            )
            old["corridor"] = (
                np.where(lat >= self.settings.divider_lat, "SH1 North", "SH1 South")
                if family == "SH1"
                else "Route 70"
            )
            old["segment_id"] = [f"{family}_{i:04d}" for i in range(len(old))]
            old["length_m"] = old.geometry.length
            roads.append(old)
            self.tiles[family] = {}
            for statusfile in sorted((base / "tile_results").glob("*/status.json")):
                status = json.loads(statusfile.read_text())
                td = statusfile.parent
                if status["state"] not in ["processed", "unobserved"]:
                    raise ValueError(f"Failed source tile: {td.name}")
                with rasterio.open(td / "probability.tif") as src:
                    assert src.crs.to_epsg() == 2193 and src.shape == (384, 384)
                    prob = src.read(1)
                    valid = (
                        (src.read_masks(1) > 0)
                        & np.isfinite(prob)
                        & (prob >= 0)
                        & (prob <= 1)
                    )
                    aff = src.transform
                    assert abs(aff.a - 10) < 1e-8 and abs(aff.e + 10) < 1e-8
                with rasterio.open(td / "binary.tif") as src:
                    binary = src.read(1)
                    assert src.transform == aff and src.crs.to_epsg() == 2193
                    assert np.array_equal(
                        valid, np.isin(binary, [0, 1])
                    ), f"Mask mismatch {td}"
                    assert np.array_equal(
                        binary[valid],
                        (prob[valid] >= status["threshold"]).astype("uint8"),
                    ), f"Probability/binary mismatch {td}"
                # Halo must be excluded: every physical core has one prediction per family.
                core = np.zeros(prob.shape, bool)
                core[64:320, 64:320] = True
                assert not (valid & ~core).any(), f"Unexpected valid halo {td}"
                assert int(valid.sum()) == status["valid_pixels"]
                entry = dict(
                    prob=prob, valid=valid, transform=aff, status=status, path=td
                )
                self.tiles[family][status["tile_id"]] = entry
                tile_rows.append(
                    dict(
                        family=family,
                        tile_id=status["tile_id"],
                        state=status["state"],
                        pre_date=status["pre"]["date"],
                        post_date=status["post"]["date"],
                        pre_scene=status["pre"]["id"],
                        post_scene=status["post"]["id"],
                        pre_cloud_screen=status["pre"].get(
                            "cloud_screen", "not recorded"
                        ),
                        post_cloud_screen=status["post"].get(
                            "cloud_screen", "not recorded"
                        ),
                        valid_pixels=int(valid.sum()),
                        coverage_pct=100 * valid.sum() / 65536,
                        original_detected_pixels=int((binary == 1).sum()),
                    )
                )
        self.roads = gpd.GeoDataFrame(pd.concat(roads, ignore_index=True), crs=CRS)
        assert self.roads.segment_id.is_unique
        self.tile_table = pd.DataFrame(tile_rows)
        self.tile_table.to_csv(self.out / "imagery_manifest.csv", index=False)
        self.road_unions = {
            c: unary_union(g.geometry) for c, g in self.roads.groupby("corridor")
        }
        self.gns = gpd.read_file(
            self.root / "data/all_corridor_GNS_sources.geojson"
        ).to_crs(CRS)
        if "objectid" in self.gns:
            self.gns = self.gns.drop_duplicates("objectid")
        self.gns = (
            self.gns[self.gns.geometry.geom_type == "Point"]
            .copy()
            .reset_index(drop=True)
        )
        return self

    def detect(self, threshold=None, save=True):
        """Join valid thresholded pixels across tile cores, then apply the area filter."""
        threshold = self.settings.threshold if threshold is None else threshold
        result = {}
        raws = {}
        for family, tiles in self.tiles.items():
            pieces = []
            for t in tiles.values():
                hit = t["valid"] & (t["prob"] >= threshold)
                pieces.extend(
                    shape(g)
                    for g, v in shapes(
                        hit.astype("uint8"),
                        mask=hit,
                        transform=t["transform"],
                        connectivity=8,
                    )
                    if v == 1
                )
            raw = (
                shapely.make_valid(unary_union(pieces))
                if pieces
                else GeometryCollection()
            )
            filtered = [
                p for p in polygon_parts(raw) if p.area >= self.settings.minimum_area_m2
            ]
            result[family] = unary_union(filtered)
            raws[family] = raw
            if save:
                self.predictions[family] = gpd.GeoDataFrame(
                    {
                        "detection_id": [
                            f"{family}_LS{i:04d}" for i in range(len(filtered))
                        ],
                        "family": [family] * len(filtered),
                        "area_m2": [p.area for p in filtered],
                    },
                    geometry=filtered,
                    crs=CRS,
                )
        if save:
            self.unions = result
            self.raw_unions = raws
            self.prediction_gdf = gpd.GeoDataFrame(
                pd.concat(self.predictions.values(), ignore_index=True), crs=CRS
            )
            self.prediction_gdf.to_crs(4326).to_file(
                self.out / "landslide_candidates.geojson", driver="GeoJSON"
            )
        return result

    def sample(self, family, point):
        tileid = f"C_{math.floor(point.x/2560)}_{math.floor(point.y/2560)}"
        t = self.tiles[family].get(tileid)
        if t is None:
            return False, None
        row, col = rasterio.transform.rowcol(t["transform"], point.x, point.y)
        if not (0 <= row < 384 and 0 <= col < 384):
            return False, t
        return bool(t["valid"][row, col]), t

    def assess_roads(self):
        """Classify every road section by candidate proximity and paired observation coverage."""
        rows = []
        for _, r in self.roads.iterrows():
            points = [
                r.geometry.interpolate(k / 10, normalized=True) for k in range(11)
            ]
            samples = [self.sample(r.family, p) for p in points]
            vis = sum(x[0] for x in samples) / 11
            pred = self.unions[r.family]
            dist = pred.distance(r.geometry) if not pred.is_empty else np.inf
            if dist <= self.settings.impact_distance_m:
                category = "potential_impact"
            elif dist <= self.settings.nearby_distance_m:
                category = "nearby_detection"
            elif vis < self.settings.minimum_visibility:
                category = "insufficient_observation"
            else:
                category = "no_detected_impact"
            # Hazard evidence and observation coverage are separate attributes.
            coverage = (
                "adequate" if vis >= self.settings.minimum_visibility else "limited"
            )
            dates = sorted(
                {
                    x[1]["status"]["post"]["date"]
                    for x in samples
                    if x[1] is not None and x[0]
                }
            )
            pre = sorted(
                {
                    x[1]["status"]["pre"]["date"]
                    for x in samples
                    if x[1] is not None and x[0]
                }
            )
            p = r.geometry.interpolate(0.5, normalized=True)
            lon, lat = WGS.transform(p.x, p.y)
            rows.append(
                dict(
                    segment_id=r.segment_id,
                    corridor=r.corridor,
                    family=r.family,
                    category=category,
                    label=CLASSES[category][0],
                    length_m=float(r.geometry.length),
                    visible_fraction=vis,
                    coverage=coverage,
                    distance_to_detection_m=float(dist) if np.isfinite(dist) else None,
                    pre_dates=", ".join(pre),
                    post_dates=", ".join(dates),
                    longitude=lon,
                    latitude=lat,
                    original_class=r.original_class,
                    geometry=r.geometry,
                )
            )
        self.assessment = gpd.GeoDataFrame(rows, crs=CRS)
        self.assessment.to_crs(4326).to_file(
            self.out / "road_assessment.geojson", driver="GeoJSON"
        )
        self.assessment.drop(columns="geometry").to_csv(
            self.out / "road_assessment.csv", index=False
        )
        oldmap = {
            "possible_impingement": "potential_impact",
            "inspection_priority": "nearby_detection",
            "unknown_insufficient_imagery": "insufficient_observation",
            "no_detected_landslide_not_safe": "no_detected_impact",
        }
        self.assessment["changed_from_presentation"] = (
            self.assessment.original_class.map(oldmap) != self.assessment.category
        )
        self.assessment[self.assessment.changed_from_presentation].drop(
            columns="geometry"
        ).to_csv(self.out / "changed_road_sections.csv", index=False)
        summary = (
            self.assessment.groupby(["corridor", "category"])
            .agg(segments=("segment_id", "size"), length_m=("length_m", "sum"))
            .reset_index()
        )
        summary["length_km"] = summary.length_m / 1000
        summary.to_csv(self.out / "road_summary.csv", index=False)
        self.road_summary = summary
        return summary

    def evaluate_points(self, unions=None, save=True):
        """Count observable GNS source points near predictions; this is not precision."""
        unions = self.unions if unions is None else unions
        names = ["SH1 North", "SH1 South", "Route 70"]
        if not hasattr(self, "reference_points"):
            pts = self.gns.copy()
            distances = np.column_stack(
                [pts.geometry.distance(self.road_unions[c]).values for c in names]
            )
            which = distances.argmin(axis=1)
            closest = distances.min(axis=1)
            pts["corridor"] = [names[i] for i in which]
            pts["distance_to_road_m"] = closest
            self.reference_points = pts[
                closest <= self.settings.corridor_buffer_m
            ].copy()
        pts = self.reference_points.copy()
        observed = []
        ds = []
        for _, r in pts.iterrows():
            family = "R70" if r.corridor == "Route 70" else "SH1"
            ob, _ = self.sample(family, r.geometry)
            observed.append(ob)
            # Match against full family polygons; avoid artificial north/south border loss.
            ds.append(
                float(unions[family].distance(r.geometry))
                if ob and not unions[family].is_empty
                else np.nan
            )
        pts["observed"] = observed
        pts["distance_to_prediction_m"] = ds
        pts["match_status"] = np.select(
            [
                ~pts.observed,
                pts.distance_to_prediction_m <= 50,
                pts.distance_to_prediction_m <= 100,
            ],
            ["unobserved", "within_50m", "within_100m"],
            default="unmatched_over_100m",
        )
        rows = []
        for name in names + ["Pooled"]:
            g = pts if name == "Pooled" else pts[pts.corridor == name]
            obs = int(g.observed.sum())
            n = len(g)
            n50 = int((g.distance_to_prediction_m <= 50).sum())
            n100 = int((g.distance_to_prediction_m <= 100).sum())
            rows.append(
                dict(
                    corridor=name,
                    total_reference_points=n,
                    observed=obs,
                    unobserved=n - obs,
                    matched_50m=n50,
                    matched_100m=n100,
                    coverage_pct=100 * obs / n if n else None,
                    match_50m_pct=100 * n50 / obs if obs else None,
                    match_100m_pct=100 * n100 / obs if obs else None,
                    all_reference_matched_50m_pct=100 * n50 / n if n else None,
                )
            )
        table = pd.DataFrame(rows)
        if save:
            self.point_results = table
            self.point_gdf = pts
            table.to_csv(self.out / "gns_point_results.csv", index=False)
            pts.to_crs(4326).to_file(
                self.out / "gns_point_assessment.geojson", driver="GeoJSON"
            )
            pts.drop(columns="geometry").to_csv(
                self.out / "gns_point_assessment.csv", index=False
            )
        return table

    def sensitivity(self):
        """Compare three thresholds without training or choosing a threshold from the references."""
        rows = []
        for threshold in [0.3, 0.5, 0.7]:
            u = (
                self.unions
                if threshold == self.settings.threshold
                else self.detect(threshold, save=False)
            )
            table = self.evaluate_points(u, save=False)
            for row in table.to_dict("records"):
                row["threshold"] = threshold
                fams = ["R70"] if row["corridor"] == "Route 70" else ["SH1"]
                if row["corridor"] == "Pooled":
                    fams = ["SH1", "R70"]
                # Area is reported for full family only in the pooled row to avoid double counting SH1.
                row["candidate_area_km2"] = (
                    sum(u[f].area for f in fams) / 1e6
                    if row["corridor"] in ["Pooled", "Route 70"]
                    else None
                )
                row["unique_candidate_area_km2"] = (
                    unary_union([u[f] for f in fams]).area / 1e6
                    if row["corridor"] in ["Pooled", "Route 70"]
                    else None
                )
                rows.append(row)
        self.sensitivity_results = pd.DataFrame(rows)
        self.sensitivity_results.to_csv(
            self.out / "threshold_sensitivity.csv", index=False
        )
        return self.sensitivity_results
