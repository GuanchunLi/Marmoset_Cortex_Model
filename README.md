# Marmoset_Cortex_Model

# README

## Overview

This repository provides the source code and data associated with our recent manuscript. It includes executable Jupyter notebooks and auxiliary Python functions necessary for reproducing all analyses, simulations, and figures presented in the manuscript.

## 1. System Requirements

### Software Dependencies
The provided code was developed and tested with the following software versions:

- **Python**: 3.9.7
- **SciPy**: 1.7.1
- **NumPy**: 1.24.3
- **pandas**: 1.3.4
- **statsmodels**: 0.12.2
- **NetworkX**: 2.6.3
- **NeuroDSP**: 2.1.0
- **FOOOF**: 1.1.0

### Operating System
- The software has been thoroughly tested on **Windows 11**.

### Hardware Requirements
- No specialized or non-standard hardware is required.

## 2. Installation Guide

### Installation Instructions
No formal installation procedure is necessary. All necessary Python scripts and auxiliary functions are provided in this repository, along with executable Jupyter notebooks. Users are expected to directly run the notebooks provided.

### Typical Installation Time
- Not applicable, as installation is not required.

## 3. Demonstration

### Running the Demo

To reproduce the results presented in the manuscript, please execute the following self-contained Jupyter notebooks provided in the repository:

- **`Marmoset_Exp_Timescale_Model_Introcution.ipynb`**: Contains code and figures for the analysis of timescales based on marmoset ECoG experimental data.

- **`Method_Fitting_Hi.ipynb`**: Illustrates the procedure used to fit the composite gradient employed in the model, utilizing experimental data.

- **`Marmoset_Model_Timescale_Result.ipynb`**: Provides code and figures for all simulations and analyses of the hierarchical organization of timescales in the multi-regional model of the marmoset cortex.

- **`Marmoset_Model_FC.ipynb`**: Includes code and figures for simulations and analyses related to functional connectivity in the multi-regional marmoset cortical model.

- **`Marmoset_Model_Signal_Prop.ipynb`**: Contains code and figures for simulations and analyses related to signal propagation within the multi-regional marmoset cortical model.

### Data Sources
The experimental data analyzed in these notebooks are sourced from:

- **Brain/MINDS Marmoset Brain ECoG Auditory Dataset 01** (Komatsu, M., Ichinohe, N., DataID: 4924). [Link to dataset](https://dataportal.brainminds.jp/ecog-auditory-01)

- **Brain/MINDS Marmoset Optogenetics Dataset 01** (Komatsu, M., Sugano, E., Tomita, H., Fujii, N., DataID: 3718). [Link to dataset](https://dataportal.brainminds.jp/ecog-optogenetics-01)

Both datasets are included under the `Data` directory. The simulated data generated from the model is located under `Data/Marmoset_Model`.

### Expected Output
- Outputs of simulations, analyses, and figures are entirely contained within each provided Jupyter notebook.

### Expected Runtime
- Most simulations and analyses should complete within a few minutes on a typical desktop computer.
- Longer runtimes may occur for notebooks involving parallel computations (`Marmoset_Model_FC.ipynb` and `Marmoset_Model_Signal_Prop.ipynb`), with runtime dependent on the available number of computational threads.

## 4. Instructions for Use

### Running the Software on Your Data
- To apply this software framework to your own datasets, execute the corresponding self-contained Jupyter notebooks directly. Each notebook clearly documents the required input format and steps necessary for conducting the simulations and analyses.
- Notebooks are self-contained, and users can run them independently in any order.

### Reproducing Manuscript Results
- All figures and results shown in the manuscript can be fully reproduced using the notebooks provided in this repository.

