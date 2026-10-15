# Wildfire Phase Detection from Sentinel-2 and Sentinel-5P

Detect and stage wildfires from public satellite data alone — no ground sensors — so
the same method works anywhere a fire is reported.

The pipeline builds a per-fire time series from two Copernicus missions, labels each
date as pre-fire, active, or post-fire, and learns a discrete representation of that
series with a Vector Quantized Variational Autoencoder (VQ-VAE) using the latent space trick, 
a form of data compression.  The learned codes then drive a phase classifier.

<img width="1127" height="244" alt="image" src="https://github.com/user-attachments/assets/7a21f171-4f08-40a1-99c1-40bb94331cc9" />

## Method

Fire perimeter products tell you where a fire burned after the fact. 
The *signal of a fire's progression* is recoverable from openly available
orbital data to track vegetation loss plus atmospheric chemistry at a cadence useful for
staging. Framing it as a sequence problem rather than a per-image classification
problem lets one model generalize across fires of very different size and geography.

## Data

| Source | Product | Bands used | Role |
| --- | --- | --- | --- |
| Sentinel-2 | L2A surface reflectance | B04 (red), B08 (NIR), SCL | NDVI, cloud masking |
| Sentinel-5P | L2 | CO, HCHO, NO₂, O₃, SO₂, CH₄ | Atmospheric signature |

Accessed through the [Copernicus Data Space](https://dataspace.copernicus.eu/) openEO
API. Study areas are defined as bounding-box polygons per fire under `experiments/`.

No imagery is committed, the notebooks fetch and cache results locally.

## Pipeline

**Sentinel-2 → NDVI time series**
1. Load L2A cube for the fire's bounding box and temporal extent.
2. Compute NDVI from infrared bands B04 and B08.
3. Build a cloud mask from the SCL scene-classification band, dilate it with an
   11×11 Gaussian kernel (σ ≈ 1.6) and a threshold, then apply it to NDVI. Dilating
   catches cloud edges and thin cirrus that the raw SCL classes miss.
4. Spatially aggregate to a single mean value per date over the fire polygon.
5. Smooth with a Savitzky–Golay filter running as a **server-side openEO UDF**, with
   forward/backward fill and a window size that adapts to the number of available
   time steps. Doing this server-side avoids pulling full cubes down to the client.
6. Export via `execute_batch` to CSV, skipping work when the output already exists.

**Sentinel-5P → gas time series**
Each of the six gases is loaded, spatially aggregated the same way, and captured
per band.

**Merge and label**
NDVI and gas series are outer-joined on date, then each date is labeled `pre`,
`active`, or `post` from the three temporal extents defined for that fire. Gaps are
linearly interpolated, then forward- and backward-filled.

**Model**
- Features are standardized, then cut into sliding windows (length 10) over the seven
  channels (NDVI + six gases).
- `model/vq_vae.py` implements a 3D convolutional VQ-VAE convoluted to 1D space:
  - Encoder: two `Conv1d` layers, hidden 256 → latent 64
  - Vector quantizer: 512-entry codebook, commitment cost 0.25, straight-through
    estimator on the backward pass
  - Decoder: mirrors the encoder back to seven channels
- Trained with MSE reconstruction loss plus the VQ commitment loss (Adam, lr 1e-3).
- Reconstructions are plotted per channel against the originals as a sanity check.

**Classification**
The quantized representation feeds a phase classifier evaluated with stratified
k-fold cross-validation, reporting per-class precision/recall/F1 and a confusion
matrix.

## Repo layout

```
flows/CSVToFeatureFlow.py   Metaflow flow: fire records (CSV) → GeoJSON polygon features
model/vq_vae.py             Encoder, VectorQuantizer, Decoder, VQVAE
utils/openeo_connection.py  Authenticated openEO connection
utils/senti2_bap.py         Best-available-pixel compositing helper
bin/senti_tools.py          Sentinel-2 index computation, masking, Savitzky–Golay smoothing
bin/plot_timeseries.py      Plotting helpers
experiments/                Per-fire bounding boxes and date ranges
experiments-multi/          Multi-fire run inputs (experiments_geo.csv)
pipeline.ipynb              Single-fire end-to-end walkthrough
multi-pipeline.ipynb        Multi-fire variant
openeo_explore.ipynb        Data exploration against the openEO catalog
main.py                     Entry point for the Metaflow flow
```

The Metaflow flow exists so the feature-generation step runs identically across many
fires instead of being hand-edited per notebook run.

## Running it

Requires Python 3.13 and a free Copernicus Data Space account.

```bash
pip install -r requirements.txt
python main.py          # run the Metaflow feature flow
jupyter lab             # then open pipeline.ipynb
```

First openEO call triggers an OIDC device-code login in the browser. Batch jobs run on
Copernicus infrastructure and can take several minutes per fire.

## Status

Archived research code from the spring 2025 course project, recommitted in October
2026 when I pulled it onto a new machine It is
exploratory, not productionized: no tests or CI.

## Attribution

Contains modified Copernicus Sentinel data (2025), processed via the Copernicus Data
Space Ecosystem openEO API. 
