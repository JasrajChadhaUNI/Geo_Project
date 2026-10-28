"""GEOG 761 | Full SH1 Kaikoura coastal corridor: genuine CPU CNN inference.

Prediction stage does not consult any GNS landslide inventory.
Input: pre/post 2016 Sentinel-2 L1C, model trained on L2A, modern OSM SH1.
Output: every corridor core has processed/unknown/failed status. No safety claims.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import LineString, Point, box, mapping, shape
from shapely.ops import transform, unary_union
from pyproj import Transformer
import torch

LOG = logging.getLogger("geog761")
BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
GEE_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
# Explicit operational corridor around 2016-damaged coastal SH1, Oaro to Clarence.
BBOX = dict(west=173.26, south=-42.66, east=174.32, north=-42.08)
CORE_M, HALO_M, RES, SIZE = 2560, 640, 10, 384
TO_NZTM = Transformer.from_crs(4326, 2193, always_xy=True)
TO_WGS = Transformer.from_crs(2193, 4326, always_xy=True)
CHECKPOINT_SHA256 = "0c16cb4353e2d5e092032f04786c8f81179b3c584d57072419c4daa7f88b79cd"


def session():
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def get_sh1(out):
    """Download *real OSM ways* with SH1 ref/name; cache to GeoJSON. No invented geometry."""
    cache = out / "01_sh1_osm_current.geojson"
    if cache.exists():
        data = json.loads(cache.read_text())
        feats = data.get("features", [])
        if not feats:
            raise RuntimeError("Cached SH1 file is empty")
        return feats
    q = (
        """[out:json][timeout:180];
    (way["highway"]["ref"~"(^|;)[ ]*(SH[ ]?)?1([ ];|$)"](%(south)s,%(west)s,%(north)s,%(east)s);
     way["highway"]["name"~"State Highway 1",i](%(south)s,%(west)s,%(north)s,%(east)s););
    out geom;"""
        % BBOX
    )
    errors = []
    payload = None
    for endpoint in [
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]:
        try:
            resp = requests.get(
                endpoint,
                params={"data": q},
                timeout=40,
                headers={"User-Agent": "GEOG761-University-Research/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            if len(data.get("elements", [])) < 3:
                raise ValueError("Too few SH1 ways returned")
            payload = data
            break
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    if payload is None:
        raise RuntimeError(
            "Could not fetch real SH1 alignment; rerun with internet/Overpass access. "
            + "; ".join(errors)
        )
    seen = set()
    feats = []
    for e in payload["elements"]:
        if e.get("type") != "way" or e.get("id") in seen:
            continue
        tags = e.get("tags", {})
        ref = str(tags.get("ref", "")).replace(" ", "").lower()
        name = str(tags.get("name", "")).lower()
        if not (ref in {"1", "sh1", "statehighway1"} or "state highway 1" in name):
            continue
        pts = [
            (p["lon"], p["lat"])
            for p in e.get("geometry", [])
            if "lat" in p and "lon" in p
        ]
        if len(pts) < 2:
            continue
        seen.add(e["id"])
        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "osm_way_id": e["id"],
                    "ref": tags.get("ref", ""),
                    "name": tags.get("name", ""),
                    "source": "modern OSM (not historical 2016)",
                },
                "geometry": mapping(LineString(pts)),
            }
        )
    if not feats:
        raise RuntimeError(
            "Overpass returned ways, but none matched SH1 explicitly; do not invent alignment."
        )
    atomic_json(cache, {"type": "FeatureCollection", "features": feats})
    return feats


def build_tiles(road_features, out, buffer_m=1000):
    lines = [transform(TO_NZTM.transform, shape(f["geometry"])) for f in road_features]
    road = unary_union(lines)
    corridor = road.buffer(buffer_m)
    x0, y0, x1, y1 = corridor.bounds
    tiles = []
    for ix in range(math.floor(x0 / CORE_M), math.floor(x1 / CORE_M) + 1):
        for iy in range(math.floor(y0 / CORE_M), math.floor(y1 / CORE_M) + 1):
            core = box(ix * CORE_M, iy * CORE_M, (ix + 1) * CORE_M, (iy + 1) * CORE_M)
            if not core.intersects(corridor):
                continue
            if core.intersection(corridor).area < 1:
                continue
            t_id = f"C_{ix}_{iy}"
            # 384 x 384 model input has 64-pixel halo each side.
            xmin = ix * CORE_M - HALO_M
            ymax = (iy + 1) * CORE_M + HALO_M
            tiles.append(
                {
                    "tile_id": t_id,
                    "ix": ix,
                    "iy": iy,
                    "xmin": xmin,
                    "ymax": ymax,
                    "core_coverage_pct": round(
                        core.intersection(corridor).area / core.area * 100, 1
                    ),
                    "core": core,
                    "input": box(xmin, ymax - RES * SIZE, xmin + RES * SIZE, ymax),
                }
            )
    tiles.sort(key=lambda t: (t["ix"], t["iy"]))
    features = [
        {
            "type": "Feature",
            "properties": {
                k: t[k] for k in ("tile_id", "ix", "iy", "core_coverage_pct")
            },
            "geometry": mapping(transform(TO_WGS.transform, t["core"])),
        }
        for t in tiles
    ]
    atomic_json(
        out / "02_full_corridor_tiles.geojson",
        {"type": "FeatureCollection", "features": features},
    )
    atomic_json(
        out / "02_full_corridor_buffer.geojson",
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"buffer_metres": buffer_m},
                    "geometry": mapping(transform(TO_WGS.transform, corridor)),
                }
            ],
        },
    )
    return tiles, road


def connect_ee(project):
    import ee

    try:
        ee.Initialize(project=project)
        ee.Number(1).getInfo()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
        ee.Number(1).getInfo()
    return ee


def scene_candidates(ee, tile_geom, label, start_date, end_date, target, limit=9):
    col = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(tile_geom)
        .filterDate(start_date, end_date)
    )

    def with_coverage(im):
        frac = (
            im.geometry()
            .intersection(tile_geom, ee.ErrorMargin(10))
            .area(10)
            .divide(tile_geom.area(10))
        )
        return im.set("tile_coverage", frac)

    col = col.map(with_coverage).filter(ee.Filter.gte("tile_coverage", 0.98))
    ids = col.aggregate_array("system:index").getInfo() or []
    times = col.aggregate_array("system:time_start").getInfo() or []
    clouds = col.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo() or []
    ref = datetime.fromisoformat(target).replace(tzinfo=timezone.utc)
    candidates = []
    for id, ms, cloud in zip(ids, times, clouds):
        d = datetime.fromtimestamp(ms / 1000, timezone.utc)
        days = abs((d - ref).total_seconds()) / 86400
        candidates.append(
            {
                "id": id,
                "date": d.date().isoformat(),
                "days_from_target": days,
                "scene_cloud_pct": float(cloud or 0),
            }
        )
    # First screen around preferred event-specific dates; then QA60 tile visibility.
    candidates.sort(key=lambda p: (p["days_from_target"], p["scene_cloud_pct"]))
    selected = []
    for c in candidates[:limit]:
        im = ee.Image(col.filter(ee.Filter.eq("system:index", c["id"])).first())
        qa = im.select("QA60")
        ok = (
            im.select("B2")
            .mask()
            .And(qa.bitwiseAnd(1 << 10).eq(0))
            .And(qa.bitwiseAnd(1 << 11).eq(0))
        ).rename("ok")
        fraction = (
            ok.unmask(0)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=tile_geom,
                scale=80,
                maxPixels=1000000,
                bestEffort=True,
            )
            .get("ok")
            .getInfo()
        )
        c["qa60_clear_fraction"] = float(fraction or 0)
        c["ranking_score"] = c["qa60_clear_fraction"] - 0.012 * c["days_from_target"]
        selected.append(c)
    selected.sort(key=lambda x: x["ranking_score"], reverse=True)
    if not selected:
        raise RuntimeError(f"No 98% covering {label} scene {start_date}..{end_date}")
    best = selected[0]
    return ee.Image(col.filter(ee.Filter.eq("system:index", best["id"])).first()), best


def fetch_raster(ee, image, meta, tile, out, label):
    from rasterio.transform import Affine

    name = hashlib.sha256(
        (
            meta["id"] + "|" + str(tile["xmin"]) + "|" + str(tile["ymax"]) + "|v3"
        ).encode()
    ).hexdigest()[:14]
    p = out / f"{label}_{name}.tif"
    expected = Affine(10, 0, tile["xmin"], 0, -10, tile["ymax"])
    if not p.exists():
        qa = image.select("QA60")
        validity = (
            image.select(GEE_BANDS)
            .mask()
            .reduce(ee.Reducer.min())
            .And(qa.bitwiseAnd(1 << 10).eq(0))
            .And(qa.bitwiseAnd(1 << 11).eq(0))
        )
        scores = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filter(
            ee.Filter.eq("system:index", meta["id"])
        )
        if scores.size().getInfo() > 0:
            validity = validity.And(ee.Image(scores.first()).select("cs_cdf").gte(0.60))
            meta["cloud_screen"] = "QA60 + Cloud Score+ cs_cdf >= .60"
        else:
            meta["cloud_screen"] = "QA60 only; cloud shadows less reliable"
        twelve = (
            image.select(GEE_BANDS, BANDS)
            .resample("bilinear")
            .divide(10000)
            .clamp(0, 1)
            .toFloat()
        )
        combined = twelve.unmask(0).addBands(
            validity.rename("valid").unmask(0).toFloat()
        )
        url = combined.getDownloadURL(
            {
                "bands": BANDS + ["valid"],
                "crs": "EPSG:2193",
                "crs_transform": [RES, 0, tile["xmin"], 0, -RES, tile["ymax"]],
                "dimensions": [SIZE, SIZE],
                "format": "GEO_TIFF",
            }
        )
        resp = session().get(url, timeout=180)
        resp.raise_for_status()
        part = p.with_suffix(".tif.part")
        part.write_bytes(resp.content)
        part.replace(p)
    try:
        with rasterio.open(p) as r:
            if (
                r.count != 13
                or r.width != SIZE
                or r.height != SIZE
                or r.crs.to_epsg() != 2193
                or not r.transform.almost_equals(expected)
            ):
                raise RuntimeError(f"Raster band/grid mismatch: {p}")
            arr = r.read().astype("float32")
    except Exception:
        p.unlink(missing_ok=True)
        raise
    return (
        np.clip(arr[:12], 0, 1),
        (arr[12] >= 0.5) & np.isfinite(arr[:12]).all(axis=0),
        p,
    )


def checkpoint_ready(repo):
    ck = repo / "checkpoints/official/unet_r50_diff.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)

    def verified():
        if not ck.exists() or ck.stat().st_size < 100_000_000:
            return False
        with ck.open("rb") as fd:
            return hashlib.sha256(fd.read()).hexdigest() == CHECKPOINT_SHA256

    if not verified():
        LOG.warning(
            "Official weights absent/invalid; downloading verified published checkpoint."
        )
        import gdown

        tmp = ck.with_suffix(".pt.part")
        if tmp.exists():
            tmp.unlink()
        gdown.download(
            id="1Z6FOqWJ2ZmiJetrP48pXZiHSzizjZKdj", output=str(tmp), quiet=False
        )
        if tmp.exists():
            tmp.replace(ck)
        if not verified():
            raise RuntimeError(
                "Pretrained weights failed saved SHA256 provenance check."
            )
    return ck


def process_tile(ee, model, tile, out, threshold=0.5):
    td = out / "tile_results" / tile["tile_id"]
    td.mkdir(parents=True, exist_ok=True)
    status_path = td / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        # Successful result is reusable only with the fixed threshold and georeferenced data.
        if (
            status.get("state") in {"processed", "unobserved"}
            and status.get("threshold") == threshold
            and (td / "binary.tif").exists()
            and (td / "probability.tif").exists()
        ):
            LOG.info("CACHED %s %s", tile["tile_id"], status["state"])
            return status
    status = {
        "tile_id": tile["tile_id"],
        "ix": tile["ix"],
        "iy": tile["iy"],
        "state": "failed",
        "threshold": threshold,
        "valid_pixels": 0,
        "predicted_pixels": 0,
        "input_format": "Sentinel-2 2016 Level-1C TOA; model trained on L2A SR",
    }
    try:
        geo = ee.Geometry.Rectangle(
            [tile["xmin"], tile["ymax"] - 3840, tile["xmin"] + 3840, tile["ymax"]],
            proj="EPSG:2193",
            geodesic=False,
        )
        before, mb = scene_candidates(
            ee, geo, "PRE", "2016-10-01", "2016-11-14", "2016-10-27"
        )
        after, ma = scene_candidates(
            ee, geo, "POST", "2016-11-14", "2016-12-09", "2016-11-23"
        )
        status.update(pre=mb, post=ma)
        a, ma_valid, pre_file = fetch_raster(ee, before, mb, tile, td, "pre")
        b, mb_valid, post_file = fetch_raster(ee, after, ma, tile, td, "post")
        valid = ma_valid & mb_valid
        core = np.zeros((SIZE, SIZE), bool)
        core[64:320, 64:320] = True
        ev = valid & core
        status["valid_pixels"] = int(ev.sum())
        # Full zero-valued/unobserved raster preserves unknown status in aggregation.
        transform_ = Affine(10, 0, tile["xmin"], 0, -10, tile["ymax"])
        profile = dict(
            driver="GTiff",
            height=SIZE,
            width=SIZE,
            count=1,
            crs="EPSG:2193",
            transform=transform_,
            compress="deflate",
        )
        if ev.sum() < 1000:
            prob = np.full((SIZE, SIZE), -9999, dtype="float32")
            binary = np.full((SIZE, SIZE), 255, dtype="uint8")
            status["state"] = "unobserved"
            status["reason"] = "Too few valid central pixels"
            status["available_clear_pixels"] = int(ev.sum())
            status["valid_pixels"] = 0
        else:
            a[:, ~valid] = 0
            b[:, ~valid] = 0
            with torch.inference_mode():
                logits = model(
                    torch.from_numpy(a[None]),
                    torch.from_numpy(b[None]),
                    torch.zeros((1, 2, SIZE, SIZE)),
                )
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                pp = torch.sigmoid(logits).cpu().numpy().squeeze()
            if pp.shape != (SIZE, SIZE) or not np.isfinite(pp).all():
                raise RuntimeError(f"Unexpected CNN output: {pp.shape}")
            hit = (pp >= threshold) & ev
            status["predicted_pixels"] = int(hit.sum())
            status["state"] = "processed"
            prob = np.where(ev, pp, -9999).astype("float32")
            binary = np.where(ev, hit.astype("uint8"), 255).astype("uint8")
        with rasterio.open(
            td / "probability.tif", "w", **profile, dtype="float32", nodata=-9999
        ) as w:
            w.write(prob, 1)
        with rasterio.open(
            td / "binary.tif", "w", **profile, dtype="uint8", nodata=255
        ) as w:
            w.write(binary, 1)
        status["processed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, status)
        return status
    except Exception as exc:
        status["reason"] = str(exc)
        status["processed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, status)
        LOG.exception("Tile %s failed; final export will stop.", tile["tile_id"])
        return status


def merge_detections(tiles, out):
    polys = []
    masks = {}
    counts = dict(processed=0, unobserved=0, failed=0)
    for t in tiles:
        td = out / "tile_results" / t["tile_id"]
        sp = td / "status.json"
        if not sp.exists():
            counts["failed"] += 1
            continue
        st = json.loads(sp.read_text())
        state = st["state"]
        counts[state] = counts.get(state, 0) + 1
        p = td / "binary.tif"
        if not p.exists():
            continue
        with rasterio.open(p) as r:
            mask = r.read(1)
            masks[(t["ix"], t["iy"])] = (mask, r.transform)
            if state != "processed":
                continue
            for g, v in shapes(
                (mask == 1).astype("uint8"),
                mask=mask == 1,
                transform=r.transform,
                connectivity=8,
            ):
                if v == 1:
                    candidate = shape(g)
                    if not candidate.is_valid:
                        candidate = candidate.buffer(0)
                    if not candidate.is_empty:
                        polys.append(candidate)
    # Adjacent tile boundaries merge without overlapping core inference and double counting.
    merged = unary_union(polys) if polys else None
    if merged is None:
        parts = []
    elif merged.geom_type == "Polygon":
        parts = [merged]
    elif merged.geom_type == "MultiPolygon":
        parts = list(merged.geoms)
    else:
        parts = [g for g in getattr(merged, "geoms", []) if g.geom_type == "Polygon"]
    parts = [
        p for p in parts if p.area >= 300
    ]  # small pixels remain in raw prediction rasters
    merged = unary_union(parts) if parts else None
    points = []
    features = []
    for n, p in enumerate(sorted(parts, key=lambda a: a.area, reverse=True), 1):
        ll = transform(TO_WGS.transform, p)
        lon, lat = TO_WGS.transform(p.centroid.x, p.centroid.y)
        prop = {
            "landslide_id": f"LS{n:04d}",
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "area_m2": round(p.area),
            "status": "unverified CNN-predicted change",
            "maps_url": f"https://www.google.com/maps?q={lat:.7f},{lon:.7f}",
        }
        features.append(
            {"type": "Feature", "properties": prop, "geometry": mapping(ll)}
        )
        points.append(prop)
    atomic_json(
        out / "03_predicted_landslides.geojson",
        {"type": "FeatureCollection", "features": features},
    )
    with (out / "03_predicted_landslides.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "landslide_id",
                "latitude",
                "longitude",
                "area_m2",
                "status",
                "maps_url",
            ],
        )
        w.writeheader()
        w.writerows(points)
    return merged, parts, masks, counts


def classify_roads(features, predicted, masks, out, segment_m=200):
    lines = [transform(TO_NZTM.transform, shape(f["geometry"])) for f in features]
    rows = []
    gfeatures = []

    def observed(x, y):
        ix = math.floor(x / CORE_M)
        iy = math.floor(y / CORE_M)
        cell = masks.get((ix, iy))
        if cell is None:
            return False
        arr, aff = cell
        col = int(math.floor((x - aff.c) / aff.a))
        row = int(math.floor((aff.f - y) / abs(aff.e)))
        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
            return arr[row, col] != 255
        return False

    # Segment each OSM way every ~200 m and sample 2016 visibility along it.
    for way_id, line in enumerate(lines):
        if line.length < 1:
            continue
        steps = max(1, math.ceil(line.length / segment_m))
        from shapely.ops import substring

        for j in range(steps):
            geom = substring(
                line, j * line.length / steps, (j + 1) * line.length / steps
            )
            sample = [geom.interpolate(k / 10, normalized=True) for k in range(11)]
            valid_frac = sum(observed(p.x, p.y) for p in sample) / len(sample)
            dist = (
                predicted.distance(geom)
                if predicted is not None and not predicted.is_empty
                else float("inf")
            )
            if dist <= 20:
                classification = "possible_impingement"
            elif dist <= 100:
                classification = "inspection_priority"
            elif valid_frac < 0.8:
                classification = "unknown_insufficient_imagery"
            else:
                classification = "no_detected_landslide_not_safe"
            info = {
                "segment_id": f"SH1_{way_id:04d}_{j:03d}",
                "class": classification,
                "length_m": round(geom.length, 1),
                "visible_fraction": round(valid_frac, 2),
                "distance_to_detection_m": (
                    None if not np.isfinite(dist) else round(float(dist), 1)
                ),
            }
            rows.append(info)
            gfeatures.append(
                {
                    "type": "Feature",
                    "properties": info,
                    "geometry": mapping(transform(TO_WGS.transform, geom)),
                }
            )
    atomic_json(
        out / "04_SH1_road_assessment.geojson",
        {"type": "FeatureCollection", "features": gfeatures},
    )
    with (out / "04_SH1_road_assessment.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "segment_id",
                "class",
                "length_m",
                "visible_fraction",
                "distance_to_detection_m",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    return rows, gfeatures
