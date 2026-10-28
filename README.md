# AI-Assisted Landslide Detection for Road Accessibility

## Project overview

This team project examines whether paired satellite images and a pretrained deep-learning model can help identify earthquake-triggered landslides near important roads. The case study focuses on the 14 November 2016 Kaikōura earthquake and three corridors in the northeastern South Island: SH1 North, SH1 South, and Inland Route 70.

The project pipeline acquires pre-event and post-event Sentinel-2 imagery, screens for clouds and invalid observations, applies a convolutional neural network (CNN) to map candidate landslide-like change, compares candidates with GNS landslide source points, and flags road sections for inspection priority. It is a retrospective screening study. It does not determine whether a road is currently open, closed, safe, or passable.

## Pipeline and notebook

The main workflow is in [`Geo_Project.ipynb`](Geo_Project.ipynb). The notebook is organised into sequential stages, with explanatory notes alongside the code:

1. Connect to Google Drive and load the project code and map assets.
2. Connect to Google Earth Engine (GEE) and find Sentinel-2 imagery for the study area.
3. Select the SH1 and Route 70 road corridors, prepare image tiles, and load the pretrained model checkpoint.
4. Screen imagery for clouds and valid common observations, then run or resume tile-level predictions.
5. Combine tile outputs into georeferenced candidate landslide polygons.
6. Divide the road network into sections and classify inspection priorities using candidate distance and image coverage.
7. Compare candidate locations with GNS source points; examine score-threshold sensitivity and repeatability.
8. Generate maps, report figures, tables, and a results archive.

The notebook uses cached predictions in Drive when available. Missing or invalid tile results can be processed from satellite imagery. Therefore, a completed run can combine reused predictions with newly processed tiles; the notebook records processing and verification information rather than treating every cached output as new inference. In the submitted run, the notebook reported 140 tiles, 979 road sections, and successful final checks.

## Model and weights

The model is **Unet-Siam-Diff with a ResNet-50 encoder**, adapted from the published bi-temporal landslide change-detection work by Monopoli et al. (2024). It compares pre-event and post-event Sentinel-2 inputs using signed after-minus-before feature differences. The model accepts 12 spectral bands and produces a per-pixel change score. This project uses no digital elevation model (DEM).

The pretrained weights were chosen so the team could apply an existing landslide change-detection approach without training a new model. The original study provides the architecture and checkpoint; it does not establish accuracy for this Kaikōura application. The checkpoint is loaded strictly and its expected SHA-256 digest is recorded in [`code/model_support.py`](code/model_support.py). The published workflow used Sentinel-2 Level-2A surface reflectance, while this project applies the model to Level-1C top-of-atmosphere imagery. This product difference is a limitation when interpreting transfer performance.

## Data and provenance

| Information | Source and role in this project |
|---|---|
| Satellite imagery | Sentinel-2 Harmonized Level-1C collection in GEE (`COPERNICUS/S2_HARMONIZED`); paired pre-event and post-event observations form the model inputs. |
| Image quality | Cloud Score+ and available quality masks support cloud/invalid-pixel screening and coverage reporting. |
| Landslide reference | GNS Kaikōura 2016 Landslide Source Location Version 2 point layer; used for 50 m and 100 m proximity comparisons. |
| Roads and geographic context | OpenStreetMap (OSM) road information and OSM context maps; the road representation is current mapping, not a verified 2016 road snapshot. |
| Model and weights | Unet-Siam-Diff / ResNet-50 architecture and pretrained checkpoint associated with Monopoli et al. (2024). The checkpoint is not bundled in this source-only package. |

The large event data and model files remain in the team's Google Drive project. The Colab notebook's `REPO` setting initially points to `/content/drive/MyDrive/GEOG761/igarss-landslide-delineation`. In that project, the checkpoint is expected at `checkpoints/official/unet_r50_diff.pt`; the existing tile predictions, masks, and source imagery are stored under `outputs/kaikoura_full_coast_SH1/` and `outputs/kaikoura_route70/`. The notebook loads these existing files when configured to resume. Google credentials are not stored in the repository.

The versioned code and fixed OSM context images are supplied separately from the large data. The notebook reads them from the `Kaikoura_GitHub_Ready.zip` uploaded to the top level of My Drive, or from an extracted folder containing `code/` and `assets/`. The three georeferenced map images are included in `assets/`; their attribution and checksums are recorded in `assets/basemap_manifest.json`.

## Results and interpretation

The saved Colab run assessed 178.38 km of road and produced 1,355 candidate landslide polygons. At the selected score threshold of 0.5, the GNS comparison included 2,564 observable points out of 2,750 total source points. Of the observable points, 1,526 (59.52%) were within 50 m of a candidate and 1,779 (69.38%) were within 100 m.

These percentages are **point-proximity agreement rates**, not overall accuracy, precision, recall, or F1. The GNS layer contains landslide source points and does not provide a complete negative sample. The notebook reports that GNS V3 polygons were not supplied for the run, so polygon-based precision, recall, F1, or intersection-over-union were not calculated. Repeatability checks compare fresh predictions with cached predictions for selected input pairs; they are not an independent accuracy assessment.

Road categories are also screening outputs. A nearby candidate can flag a section for inspection, while “no detected impact” means only that no candidate was mapped within the chosen distance under the available image conditions. It is not evidence that the road is open. The notebook examines alternative model score thresholds because the number and area of candidate detections change with this setting. The analysis does not prove that any tested threshold is optimal.

Agricultural disturbance, including ploughed fields, can resemble landslide change and remains a known source of false detections. Other limitations include cloud and shadow, image timing, Sentinel-2 spatial resolution, the L1C/L2A product difference, and the use of current road mapping. Independent polygon references, reviewed false-positive and false-negative examples, and dated agency road-status records would strengthen future validation and operational use.

## Included outputs

The `results/figures/` folder contains PNG figures from the saved Colab run so that the project outputs can be reviewed without executing the notebook. They include the pipeline diagram, road corridor maps, GNS point comparison, score-threshold sensitivity, road-threshold sensitivity, observation timing, study-area overview, spatial validation map, and an illustrative SH1 satellite example. OSM-derived maps include attribution to OpenStreetMap contributors under the Open Database License (ODbL).

When the notebook is run in Colab, its generated tables and figures are written to:

`/content/drive/MyDrive/GEOG761/igarss-landslide-delineation/outputs/final_report_clean/results/`

The final notebook cell checks required outputs and creates a timestamped results ZIP under `outputs/final_report_clean/`. The full imagery caches and model checkpoint are intentionally kept in Drive rather than copied into this repository.

## Code guide

| File | Purpose |
|---|---|
| [`code/ee_runner.py`](code/ee_runner.py) | Sentinel-2 scene selection, tile planning, imagery preparation, and model inference. |
| [`code/drive_workflow.py`](code/drive_workflow.py) | Drive cache recovery, resume/rebuild workflow, input-scene recovery, and repeatability checks. |
| [`code/model_support.py`](code/model_support.py) | Model architecture and strict pretrained-checkpoint loading. |
| [`code/analysis.py`](code/analysis.py) | Prediction assembly, candidate polygons, road categories, GNS point comparisons, and sensitivity analyses. |
| [`code/validation.py`](code/validation.py) | Optional polygon-inventory evaluation when appropriate reference layers and coverage data are supplied. |
| [`code/report_outputs.py`](code/report_outputs.py) | Report tables, figures, and workflow diagram. |
| [`code/evidence.py`](code/evidence.py) | Selected exact-input image examples for inspection. |
| [`code/basemaps.py`](code/basemaps.py) | Static and interactive maps using the included OSM context images. |

The package also includes the cleaned Colab notebook, `requirements-colab.txt`, map assets, cleanup notes, and a record of local code checks in `docs/`.

## AI assistance statement

OpenAI GPT-6 (Codex) was used as an AI assistant to help review, debug, refactor, and document parts of the project code. The AI assistant did not produce the satellite imagery or GNS reference data. The project team reviewed and ran the Colab workflow and remains responsible for the code, results, interpretation, and submitted report.

## References and data links

Google Earth Engine. (n.d.-a). *Harmonized Sentinel-2 MSI: MultiSpectral Instrument, Level-1C (TOA).* https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED

Google Earth Engine. (n.d.-b). *Cloud Score+ S2_HARMONIZED V1.* https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED

GNS Science, University of Canterbury, & Massey University. (n.d.). *Kaikōura 2016 landslide source location V2* [GIS layer]. https://gis.gns.cri.nz/server/rest/services/EILD/EILD_V2_Map/MapServer/0

Monopoli, T., Montello, F., & Rossi, C. (2024). *Landslide mapping from Sentinel-2 imagery through change detection.* arXiv. https://doi.org/10.48550/arXiv.2405.20161

OpenStreetMap contributors. (n.d.). *Copyright and licence.* https://www.openstreetmap.org/copyright
