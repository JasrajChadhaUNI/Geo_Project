"""Fixed, attributed OSM map context; no requests to the public tile server.
The assets are separate from the historical imagery used by the CNN.
"""

from pathlib import Path
import hashlib, json
import numpy as np
import rasterio
from rasterio.warp import (
    reproject,
    calculate_default_transform,
    Resampling,
    transform_bounds,
)


def load_basemaps(asset_folder):
    folder = Path(asset_folder)
    records = json.loads((folder / "basemap_manifest.json").read_text())
    layers = []
    for record in records:
        path = folder / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Basemap asset checksum mismatch: {path.name}")
        with rasterio.open(path) as source:
            if source.count != 4 or source.crs.to_epsg() != 2193:
                raise ValueError("Expected an NZTM RGBA context raster.")
            data = source.read()
            bounds = source.bounds
            layers.append(
                dict(record, rgba=np.moveaxis(data, 0, -1), bounds=bounds, path=path)
            )
    return layers


def draw_basemap(ax, layers, roads, corridor=None):
    """Place the fixed context under routes; never treat a server error image as a map."""
    x0, y0, x1, y1 = roads.total_bounds
    pad = 0.08 * max(x1 - x0, y1 - y0, 3000)
    selected = [r for r in layers if r["corridor"] == corridor] if corridor else layers
    if not selected:
        raise ValueError(f"No context raster for {corridor}")
    for layer in selected:
        left, bottom, right, top = layer["bounds"]
        ax.imshow(
            layer["rgba"],
            extent=(left, right, bottom, top),
            origin="upper",
            alpha=0.75,
            zorder=0,
        )
    ax.set(xlim=(x0 - pad, x1 + pad), ylim=(y0 - pad, y1 + pad), aspect="equal")


def interactive_preview(p, layers):
    """Folium preview uses embedded context images, without Leafmap/ipyvuetify.
    Browser access to Leaflet's JS/CSS CDN is still required for interaction.
    """
    import folium

    roads = p.assessment.to_crs(4326)
    xmin, ymin, xmax, ymax = roads.total_bounds
    m = folium.Map(
        location=[(ymin + ymax) / 2, (xmin + xmax) / 2],
        zoom_start=9,
        tiles=None,
        control_scale=True,
    )
    credit = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> | fixed current context'
    for layer in layers:
        with rasterio.open(layer["path"]) as source:
            dst, w, h = calculate_default_transform(
                source.crs, 3857, source.width, source.height, *source.bounds
            )
            data = np.zeros((4, h, w), dtype="uint8")
            for i in range(4):
                reproject(
                    source.read(i + 1),
                    data[i],
                    src_transform=source.transform,
                    src_crs=source.crs,
                    dst_transform=dst,
                    dst_crs=3857,
                    resampling=Resampling.bilinear,
                )
            b = rasterio.transform.array_bounds(h, w, dst)
            west, south, east, north = transform_bounds(3857, 4326, *b)
        folium.raster_layers.ImageOverlay(
            np.moveaxis(data, 0, -1),
            bounds=[[south, west], [north, east]],
            name=layer["corridor"] + " OSM context",
            attr=credit,
            opacity=0.85,
            control=False,
        ).add_to(m)
    from analysis import CLASSES

    for corridor, g in roads.groupby("corridor"):
        cols = ["segment_id", "category", "label", "visible_fraction", "geometry"]
        folium.GeoJson(
            json.loads(g[cols].to_json()),
            name=corridor,
            style_function=lambda f: dict(
                color=CLASSES[f["properties"]["category"]][1], weight=4
            ),
            tooltip=folium.GeoJsonTooltip(
                fields=["segment_id", "label", "visible_fraction"],
                aliases=["Section", "Screening", "Observed fraction"],
            ),
        ).add_to(m)
    if not p.prediction_gdf.empty:
        folium.GeoJson(
            json.loads(p.prediction_gdf.to_crs(4326).to_json()),
            name="Candidate landslides",
            show=False,
            style_function=lambda f: dict(color="#c44732", weight=1, fillOpacity=0.25),
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[ymin, xmin], [ymax, xmax]])
    legend = '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:12px;border:1px solid #aaa;font-size:12px"><b>Inspection priorities</b><br>'
    for label, color in CLASSES.values():
        legend += f'<span style="color:{color}">━━</span> {label}<br>'
    legend += "No detected impact ≠ confirmed open</div>"
    m.get_root().html.add_child(folium.Element(legend))
    return m
