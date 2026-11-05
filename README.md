# Where Do Frozen Foundation Time Series Models Lose the Trend?

Code for the MSc thesis of Ties Dilling (University of Twente, 2026). Four frozen foundation models (TimesFM 2.5, Moirai 2.0, Chronos-2 and Chronos-Bolt) forecast a random walk with drift, and their forecasts are compared with the best possible forecast.

## Files

- `data_generator.py`, `dataset.py`, `metrics.py`: helper code used by the notebooks
- `Dataset_Builder.ipynb`: generates the random walks
- `Chronos2.ipynb`, `Chronos_BOLT.ipynb`, `MOIRAI.ipynb`, `TimesFM.ipynb`: run the models
- `downstream.ipynb`, `excess_risk.ipynb`, `rmse_table.ipynb`, `trajectories_wins.ipynb`: make the tables and figures

## How to run

1. Set the folder paths in the notebooks (marked with `*/`).
2. Run `Dataset_Builder.ipynb`.
3. Run the four model notebooks.
4. Run the four analysis notebooks.

To only redo the analysis, download the predictions from Zenodo (data will be uploaded soon) and start at step 4.

Requirements: Python with numpy, pandas and matplotlib, plus `chronos-forecasting`, `uni2ts` and `timesfm` for the models.

## Data

The model predictions will be uploaded on Zenodo: DOI will be added after upload.

