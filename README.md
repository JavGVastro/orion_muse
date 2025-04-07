# orion_muse

This repository contains a modular, notebook-driven pipeline for processing and analyzing observational data of the Orion Nebula, focusing on radial velocity fields and structure function analysis using MUSE and related datasets.

All steps are automated using `papermill` and `Makefile` rules, with parameters managed dynamically through a centralized `params.py` file.

## Project Structure

```
orion_muse/
├── observations/              # Raw input .fits data
├── velocity_fields_maps/      # Radial velocity maps and mask creation
├── structure_function/        # Structure function computation
├── confidence_intervals/      # Monte Carlo sampling and uncertainty estimation
├── py_modules/                # Custom Python modules (e.g., rebin_utils)
├── results_files/             # (Optional) Centralized result storage
├── figures/                   # (Optional) Plot outputs for publication
├── params.py                  # Central parameter file per dataset
├── Makefile_mask_bin          # For velocity and mask generation
├── Makefile_strucfunc         # For structure function analysis
├── Makefile_confint           # For confidence interval computation
```

## Pipeline Overview

Each step of the pipeline corresponds to a Jupyter notebook template. The pipeline is run by executing these notebooks per dataset via `make` and `papermill`.

### 1. Velocity Field and Mask Creation

- Folder: `velocity_fields_maps/`
- Template: `notebook_template_mask_bin_n.ipynb`
- Command:

    make -f Makefile_mask_bin

### 2. Structure Function Analysis

- Folder: `structure_function/`
- Template: `notebook_template_strucfunc.ipynb`
- Command:

    make -f Makefile_strucfunc

### 3. Confidence Interval Estimation

- Folder: `confidence_intervals/`
- Template: `notebook_template_confint.ipynb`
- Command:

    make -f Makefile_confint

## Dynamic Parameters

All dataset-specific parameters (such as `bins`, `flux_thresh`, `sigma_thresh`, and descriptive labels like `"Orion Ha"`) are stored in `params.py`.

This script defines:

- Which datasets are processed
- Parameter values for each
- The formatted `name` parameter (e.g., `H_I-6563_mask_bin_4`)

## Dependencies

Required tools and libraries:

- Python 3.x (Anaconda recommended)
- Jupyter Notebook / JupyterLab
- `papermill`
- `matplotlib`, `numpy`, `astropy`, `scipy`
- `PyPDF2` (optional, for PDF output)

Install `papermill` via pip:

    pip install papermill

## Output Files

Each notebook execution creates:

- A result notebook: `<name>_<step>.ipynb`
- A `.json` file with processed results
- Figures and plots (e.g., in `confidence_intervals/Imgs/`)

## How to Use

From the root of the project:

    make -f Makefile_mask_bin
    make -f Makefile_strucfunc
    make -f Makefile_confint

Each stage executes notebooks for all datasets using correct parameters from `params.py`.

## Author

Created by [@JavGVastro](https://github.com/JavGVastro).  
Please open an issue or fork the repository if you'd like to contribute or extend this work.
