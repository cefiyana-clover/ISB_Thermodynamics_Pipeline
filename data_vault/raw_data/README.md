# Raw Empirical Data Vault

Due to GitHub's file size restrictions, the massive raw empirical matrices (> 100 MB) used in this pipeline are hosted externally on Zenodo to ensure repository stability, permanent DOI archiving, and absolute computational reproducibility.

## Required Files for Script 01:
To execute `01_offline_empirical_ingestion.py`, you must download the following raw datasets and place them directly into this `raw_data/` directory. 

All files are permanently archived and accessible via our Zenodo repository:
**DOI:** [10.5281/zenodo.20229135](https://doi.org/10.5281/zenodo.20229135)

Please download the following specific files from the Zenodo link above:

1. **AHBA Transcriptomic Matrix (.mat):**
   * File Name: `ROIxGene_aparcaseg_INT.mat`
   * Size: ~150 MB

2. **Neuromaps mGluR5 PET Density (.csv):**
   * File Name: `mGluR5_PET_Beliveau_Destrieux.csv`

## Execution Protocol
Once the files are downloaded and placed in this directory, the deterministic pipeline in Script 01 will automatically ingest, normalize, and lock the spatiotemporal anchors into the 6-layer architecture.
