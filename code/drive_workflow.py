"""Drive/Earth Engine orchestration. Resume caches; never invent missing observations."""

from pathlib import Path
import json, shutil, hashlib, time
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, mapping
from shapely.ops import unary_union
from pyproj import Transformer

FOLDERS = {"SH1": "kaikoura_full_coast_SH1", "R70": "kaikoura_route70"}


def route70(runner, out):
    path = out / "01_route70_osm_current.geojson"
    if path.exists():
        features = json.loads(path.read_text())["features"]
    else:
        wgs = Transformer.from_crs(2193, 4326, always_xy=True)
        points = [
            (173.64, -42.445),
            wgs.transform(1629886, 5303395),
            (173.230, -42.468),
            (173.041, -42.655),
        ]
        coords = ";".join(f"{x:.6f},{y:.6f}" for x, y in points)
        errors = []
        route = None
        for base in [
            "https://router.project-osrm.org/route/v1/driving/",
            "https://routing.openstreetmap.de/routed-car/route/v1/driving/",
        ]:
            try:
                response = runner.session().get(
                    base + coords,
                    params={
                        "overview": "full",
                        "geometries": "geojson",
                        "steps": "true",
                        "alternatives": "false",
                    },
                    timeout=90,
                )
                response.raise_for_status()
                data = response.json()
                if data.get("code") != "Ok":
                    raise ValueError("Router returned no valid route")
                snaps = [float(x.get("distance", 0)) for x in data["waypoints"]]
                if len(snaps) != 4 or max(snaps) > 5000:
                    raise ValueError("Waypoint snapping exceeds 5 km")
                candidate = data["routes"][0]
                inland = sum(
                    float(s.get("distance", 0))
                    for leg in candidate["legs"]
                    for s in leg["steps"]
                    if "inland" in s.get("name", "").lower()
                )
                if inland < 25000:
                    raise ValueError("Route does not follow enough of Inland Road")
                line = LineString(candidate["geometry"]["coordinates"])
                km = gpd.GeoSeries([line], crs=4326).to_crs(2193).length.iloc[0] / 1000
                if not 75 <= km <= 140:
                    raise ValueError(f"Unexpected Route70 length: {km:.1f} km")
                route = candidate
                break
            except Exception as e:
                errors.append(str(e))
        if route is None:
            raise RuntimeError("Route70 acquisition failed: " + "; ".join(errors))
        features = [
            {
                "type": "Feature",
                "properties": {
                    "name": "Inland Route 70",
                    "source": "Current OSM routing via OSRM; not verified 2016 alignment",
                },
                "geometry": route["geometry"],
            }
        ]
        runner.atomic_json(path, {"type": "FeatureCollection", "features": features})
    roads = gpd.GeoDataFrame.from_features(features, crs=4326).to_crs(2193)
    if roads.empty or not 75 <= roads.length.sum() / 1000 <= 140:
        raise ValueError("Invalid cached Route70 geometry")
    return features


def prepare(runner, repo, run, rebuild=False):
    """Fresh mode writes a separate tree; resume mode retains original Drive tile caches."""
    source = run / "fresh_pipeline" if rebuild else repo
    source.mkdir(exist_ok=True, parents=True)
    plans = {}
    for family, folder in FOLDERS.items():
        out = source / "outputs" / folder
        out.mkdir(exist_ok=True, parents=True)
        roadname = (
            "01_sh1_osm_current.geojson"
            if family == "SH1"
            else "01_route70_osm_current.geojson"
        )
        if (
            rebuild
            and not (out / roadname).exists()
            and (repo / "outputs" / folder / roadname).exists()
        ):
            shutil.copy2(repo / "outputs" / folder / roadname, out / roadname)
        features = runner.get_sh1(out) if family == "SH1" else route70(runner, out)
        tiles, road = runner.build_tiles(features, out, buffer_m=1000)
        assert tiles, f"No {family} tiles planned"
        plans[family] = {"out": out, "features": features, "tiles": tiles, "road": road}
        print(
            f"{family}: {len(tiles)} tiles, {road.length/1000:.1f} km road geometry",
            flush=True,
        )
    return source, plans


def run_tiles(runner, ee, model, plans):
    execution = []
    for family, plan in plans.items():
        results = []
        for i, tile in enumerate(plan["tiles"], 1):
            td = plan["out"] / "tile_results" / tile["tile_id"]
            previous = (
                json.loads((td / "status.json").read_text())
                if (td / "status.json").exists()
                else {}
            )
            complete = (
                previous.get("state") in {"processed", "unobserved"}
                and previous.get("threshold") == 0.5
                and (td / "probability.tif").exists()
                and (td / "binary.tif").exists()
            )
            status = runner.process_tile(ee, model, tile, plan["out"], threshold=0.5)
            execution.append(
                dict(
                    family=family,
                    tile_id=tile["tile_id"],
                    state=status["state"],
                    action="reused" if complete else "computed",
                )
            )
            results.append(status)
            runner.atomic_json(
                plan["out"] / "progress.json",
                {
                    "tiles_reached": i,
                    "planned": len(plan["tiles"]),
                    "failed": sum(s["state"] == "failed" for s in results),
                },
            )
            print(
                f'{family} {i}/{len(plan["tiles"])} {tile["tile_id"]}: {status["state"]}',
                flush=True,
            )
        failed = [s for s in results if s["state"] == "failed"]
        if failed:
            raise RuntimeError(
                f'{family}: {len(failed)} failed tiles; rerun this cell to retry. First error: {failed[0].get("reason")}. No final map is exported from this incomplete pass.'
            )
        merged, parts, masks, counts = runner.merge_detections(
            plan["tiles"], plan["out"]
        )
        # The same >=300 m2 union is passed to road classification and exported maps.
        runner.classify_roads(
            plan["features"], merged, masks, plan["out"], segment_m=200
        )
        runner.atomic_json(
            plan["out"] / "00_summary.json",
            {
                "tile_count": len(plan["tiles"]),
                "counts": counts,
                "reported_landslide_polygons": len(parts),
                "threshold": 0.5,
            },
        )
    return execution


def recover_inputs(runner, ee, plans, all_inputs=True):
    """Recover exact recorded scenes, not newly selected substitutes for cached predictions."""
    rows = []
    cases = {"C_627_2062", "C_633_2069", "C_636_2071"}
    for family, plan in plans.items():
        selected = (
            plan["tiles"]
            if all_inputs
            else [
                t
                for i, t in enumerate(plan["tiles"])
                if (family == "R70" and t["tile_id"] in cases)
                or (family == "SH1" and i < 3)
            ]
        )
        for i, t in enumerate(selected, 1):
            td = plan["out"] / "tile_results" / t["tile_id"]
            status = json.loads((td / "status.json").read_text())
            for phase in ["pre", "post"]:
                matches = list(td.glob(phase + "_*.tif"))
                if len(matches) > 1:
                    raise RuntimeError(
                        f"Ambiguous {phase} inputs at {td}; inspect before running."
                    )
                if len(matches) == 1:
                    rows.append(
                        {
                            "family": family,
                            "tile_id": t["tile_id"],
                            "phase": phase,
                            "state": "existing",
                        }
                    )
                    continue
                meta = dict(status[phase])
                image = ee.Image("COPERNICUS/S2_HARMONIZED/" + meta["id"])
                runner.fetch_raster(ee, image, meta, t, td, phase)
                rows.append(
                    {
                        "family": family,
                        "tile_id": t["tile_id"],
                        "phase": phase,
                        "state": "recovered_from_EE",
                        "scene": meta["id"],
                    }
                )
                print(
                    f'{family} recovered {phase}: {t["tile_id"]} ({i}/{len(selected)})',
                    flush=True,
                )
            # Check every recovered/existing pair against the saved output grid and mask.
            masks = []
            for phase in ["pre", "post"]:
                path = next(td.glob(phase + "_*.tif"))
                with rasterio.open(path) as src:
                    expected = rasterio.Affine(10, 0, t["xmin"], 0, -10, t["ymax"])
                    if (
                        src.count != 13
                        or src.shape != (384, 384)
                        or src.crs.to_epsg() != 2193
                        or src.transform != expected
                    ):
                        raise RuntimeError(f"Input grid mismatch at {path}")
                    a = src.read()
                masks.append((a[12] >= 0.5) & np.isfinite(a[:12]).all(axis=0))
            if status["state"] == "processed":
                observed = masks[0] & masks[1]
                core = np.zeros((384, 384), bool)
                core[64:320, 64:320] = True
                with rasterio.open(td / "binary.tif") as src:
                    saved = src.read(1) != 255
                if not np.array_equal(observed & core, saved):
                    raise RuntimeError(
                        f'{family}/{t["tile_id"]}: recovered imagery masks differ from cached predictions. Set REBUILD_FROM_EE=True and rerun from setup.'
                    )
    return rows


def verify_inference(p, model, all_pairs=False):
    """Fresh inference on both families; compare only cached processed pixels."""
    import torch

    rows = []
    for family, tiles in p.tiles.items():
        candidates = [
            (k, t)
            for k, t in tiles.items()
            if t["status"]["state"] == "processed"
            and len(list(t["path"].glob("pre_*.tif"))) == 1
            and len(list(t["path"].glob("post_*.tif"))) == 1
        ]
        candidates = sorted(
            candidates, key=lambda x: -int((x[1]["prob"][x[1]["valid"]] >= 0.5).sum())
        )
        if not all_pairs:
            candidates = candidates[:3]
        if not candidates:
            raise RuntimeError(
                f"No {family} image pairs available. Run input recovery."
            )
        for name, t in candidates:
            arrays = []
            masks = []
            for phase in ["pre", "post"]:
                path = next(t["path"].glob(phase + "_*.tif"))
                with rasterio.open(path) as src:
                    assert (
                        src.count == 13
                        and src.crs.to_epsg() == 2193
                        and src.transform == t["transform"]
                    )
                    a = src.read().astype("float32")
                assert (
                    np.isfinite(a[:12]).all()
                    and a[:12].min() >= 0
                    and a[:12].max() <= 1
                )
                arrays.append(a[:12])
                masks.append(a[12] >= 0.5)
            valid = masks[0] & masks[1]
            core = np.zeros(valid.shape, bool)
            core[64:320, 64:320] = True
            if not np.array_equal(valid & core, t["valid"]):
                raise RuntimeError(
                    f"{family}/{name}: retrieved masks differ from cached results. Set REBUILD_FROM_EE=True and rerun from setup to build a consistent new run."
                )
            for a in arrays:
                a[:, ~valid] = 0
            with torch.inference_mode():
                logits = model(
                    torch.from_numpy(arrays[0][None]),
                    torch.from_numpy(arrays[1][None]),
                    torch.zeros((1, 2, 384, 384)),
                )
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                prob = torch.sigmoid(logits).cpu().numpy().squeeze()
            mask = t["valid"]
            mae = float(np.abs(prob[mask] - t["prob"][mask]).mean())
            agreement = float(((prob[mask] >= 0.5) == (t["prob"][mask] >= 0.5)).mean())
            if mae > 0.001 or agreement < 0.99:
                raise RuntimeError(
                    f"{family}/{name}: fresh model diverges from cache; use REBUILD_FROM_EE=True. MAE={mae}, agreement={agreement}"
                )
            rows.append(
                {
                    "family": family,
                    "tile_id": name,
                    "MAE": mae,
                    "binary_agreement_pct": agreement * 100,
                }
            )
            print(
                f"Fresh inference {family}/{name}: {agreement:.4%} agreement",
                flush=True,
            )
    return rows


def load_gns(runner, repo, run, plans):
    dest = run / "data/all_corridor_GNS_sources.geojson"
    dest.parent.mkdir(exist_ok=True, parents=True)
    cached = (
        repo / "outputs/final_presentation_figures/all_corridor_GNS_sources.geojson"
    )
    if not dest.exists() and cached.exists():
        shutil.copy2(cached, dest)
    if not dest.exists():
        roi = unary_union([p["road"] for p in plans.values()]).buffer(1200)
        extent = ",".join(str(x) for x in roi.bounds)
        url = "https://gis.gns.cri.nz/server/rest/services/EILD/EILD_V2_Map/MapServer/0/query"
        features = []
        for offset in range(0, 100000, 1000):
            resp = runner.session().get(
                url,
                params={
                    "where": "1=1",
                    "geometry": extent,
                    "geometryType": "esriGeometryEnvelope",
                    "inSR": 2193,
                    "outSR": 4326,
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": "objectid,source_id",
                    "orderByFields": "objectid ASC",
                    "returnGeometry": "true",
                    "resultRecordCount": 1000,
                    "resultOffset": offset,
                    "f": "geojson",
                },
                timeout=90,
            )
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            batch = payload.get("features", [])
            features.extend(batch)
            if len(batch) < 1000 and not payload.get("exceededTransferLimit", False):
                break
        else:
            raise RuntimeError(
                "GNS pagination limit reached; refusing truncated inventory"
            )
        if not features:
            raise RuntimeError("No GNS reference sources returned")
        runner.atomic_json(dest, {"type": "FeatureCollection", "features": features})
    return dest
