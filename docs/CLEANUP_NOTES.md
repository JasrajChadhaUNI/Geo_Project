# Cleanup of actualfinal.ipynb

The original notebook and its HTML export were inspected without modifying either source file.

## Confirmed issues

- The static map outputs contained OSM “403 Access blocked” imagery. They were not valid basemaps despite the figure files existing.
- The interactive cell failed with `AttributeError: module 'ipyvuetify' has no attribute 'ExpansionPanelHeader'` after installing Leafmap during the run.
- The scientific-figure cell called a general plotting function that created a plain road map and pipeline diagram, then later cells replaced those files. The road-threshold figure was also displayed again later. These were functional duplication rather than identical duplicate cells.
- The code-loading cell depended on an old results ZIP or the existing Drive code folder, making it unclear which helper version a teammate would execute.
- The candidate-area sensitivity table summed the two processing families, including any overlapping footprint.

## Changes

- Supplied three small, georeferenced OSM context rasters with checksums and attribution. Static mapping and the Folium preview use these assets; no public tile requests are made from the notebook.
- Replaced Leafmap/ipyvuetify with pinned Folium 0.20.0, installed at the start. The preview needs browser JavaScript/CSS access but not live map tiles.
- Removed redundant plot creation/overwriting and repeated display. Exported the three routes as separate readable figures.
- Preserved the useful road-threshold sensitivity analysis: it measures an operational effect distinct from GNS source-point sensitivity.
- Preserved the additional SH1 exact-input evidence cell and the three Route 70 cases.
- Added ordered Markdown explanations for all stages. Kept the original Drive path and Earth Engine project default visible and editable.
- Replaced the ambiguous source bootstrap with the matching code/assets package.
- Used the compact, strictly checked inference-only model implementation. Local fresh Route 70 checks reproduce the original predictions.
- Changed the displayed candidate-area sensitivity column to a spatial union that does not double-count family overlap; the older summed column remains in the detailed analysis CSV for traceability.
- Optional polygon metrics now require documented inventory coverage as well as source/debris polygons. No polygon scores are fabricated when reference inputs are absent.
- Cleared all notebook outputs and execution counts. The cleaned file must be run before claiming its displayed outputs as a successful Colab execution.
- Added README, dependency list, Git ignore rules and test-scope documentation. No GitHub repository has been published.

## Remaining scientific limitations

The cleanup fixes execution/presentation issues, not the field false-positive problem. Source-point matching does not measure precision or prove road access. The score threshold is not independently calibrated; training/application imagery products differ; satellite observation dates vary; OSM context/geometry are current. These limits should remain explicit in the report.
