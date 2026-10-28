"""Illustrative imagery from the exact cached model inputs, not a replacement basemap."""

import numpy as np
import rasterio
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from report_outputs import save_figure
from analysis import dump


def evidence(p):
    """Exact input imagery, not replacement mosaics. Cases are illustrative, not a random accuracy sample."""
    out = p.out / "evidence"
    out.mkdir(exist_ok=True)
    selected = [
        ("C_627_2062", "Illustrative agricultural change"),
        ("C_633_2069", "Road-adjacent candidate detections"),
        ("C_636_2071", "Contrast case: inspect detections and reference misses"),
    ]
    cases = []
    for tileid, title in selected:
        t = p.tiles["R70"].get(tileid)
        if not t:
            continue
        files = [list(t["path"].glob(phase + "_*.tif")) for phase in ["pre", "post"]]
        if any(len(x) != 1 for x in files):
            continue
        arr = []
        for fl in files:
            with rasterio.open(fl[0]) as src:
                a = src.read()
            arr.append(
                np.moveaxis(
                    np.clip(a[[3, 2, 1], 64:320, 64:320] / 0.30, 0, 1) ** (1 / 2.2),
                    0,
                    -1,
                )
            )
        aff = t["transform"] * rasterio.Affine.translation(64, 64)
        extent = (aff.c, aff.c + 2560, aff.f - 2560, aff.f)
        view = box(extent[0], extent[2], extent[1], extent[3])
        roads = p.assessment[p.assessment.intersects(view)]
        preds = p.predictions["R70"]
        preds = preds[preds.intersects(view)]
        refs = p.point_gdf[p.point_gdf.intersects(view)]
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
        for i, ax in enumerate(axs):
            ax.imshow(arr[min(i, 1)], extent=extent, origin="upper")
            if len(roads):
                roads.plot(ax=ax, color="#f8df80", linewidth=1)
            if i == 2:
                if len(preds):
                    preds.boundary.plot(ax=ax, color="#ff5b62", linewidth=1)
                if len(refs):
                    refs.plot(ax=ax, color="#29e3dc", markersize=12, marker="+")
                unknown = np.ma.masked_where(
                    t["valid"][64:320, 64:320], np.ones((256, 256))
                )
                ax.imshow(
                    unknown,
                    extent=extent,
                    cmap="Greys",
                    vmin=0,
                    vmax=1,
                    alpha=0.35,
                    origin="upper",
                )
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                [
                    f"Before: {t['status']['pre']['date']}",
                    f"After: {t['status']['post']['date']}",
                    "Candidates + GNS source points",
                ][i],
                fontsize=10,
            )
        fig.suptitle(
            f"{title}\n{tileid} · exact model inputs", fontsize=12, fontweight="bold"
        )
        axs[2].legend(
            handles=[
                Line2D([0], [0], color="#ff5b62", label="CNN boundary"),
                Line2D(
                    [0],
                    [0],
                    marker="+",
                    color="#29e3dc",
                    linestyle="",
                    label="GNS source",
                ),
            ],
            loc="lower right",
            fontsize=7,
            framealpha=0.9,
        )
        for ax in axs:
            ax.set_xlabel("")
            ax.set_ylabel("")
        save_figure(fig, out, tileid)
        cases.append(
            dict(
                tile_id=tileid,
                title=title,
                pre=t["status"]["pre"]["date"],
                post=t["status"]["post"]["date"],
                caption="Illustrative case, not a labelled accuracy sample. Yellow: road. Coral: CNN boundary. Cyan: GNS source. Grey: excluded pixels.",
                image=str((out / (tileid + ".png")).relative_to(p.out)),
            )
        )
    dump(p.out / "evidence_cases.json", cases)
    return cases
