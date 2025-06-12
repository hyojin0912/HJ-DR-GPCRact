# DR-GPCRact: Interpretable prediction of ligand-induced GPCR activity

## Overview
This repository contains code and data for the study *"Interpretable prediction of ligand-induced GPCR activity via structure-based differential residue modeling"*, submitted to the *Nature Communications*. The work introduces GPCRactDB, a curated dataset of 202,925 ligand-GPCR interactions, and proposes Differential Residues (DRs)—residues with ligand-induced conformational changes—as interpretable features for predicting GPCR activity.


### Key Findings
- **GPCR-specific DR patterns**: DRs reflect structural variability across GPCRs; some remain rigid while others exhibit strong dynamic shifts.
- **High DR consistency**: DRs are conserved across ligands for the same GPCR (consistency score > 0.8).
- **Robust interpretable model**: Outperforms sequence-based deep learning models ([DeepREAL](https://academic.oup.com/bioinformatics/article/38/9/2561/6547052), [AiGPro](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00945-7)) under scaffold-based splits.
- **Generalizability**: Applicable to data-scarce GPCRs using EGNN-predicted DRs (AUROC > 0.8).
- **Experimental relevance**: DRs align with biologically validated residues in HCAR2, MLNR, and HTR2B.

## Repository Structure
- **`data/`**: Contains key data files required for model training and prediction.
  - `splits/`: Train and test sets from pair and scaffold splits (InChIKey | UniProt accession | Label).
- **`model/`**: Contains the pre-trained Keras model for immediate use.
- **`scripts/`**: Core analysis scripts.
  - `data_preparation/`: Methods to build GPCRactDB.
    - `parse_pubchem_bioassay.py`: Collects and standardizes PubChem BioAssay data.
    - `parse_other_databases.py`: Handles other database parsing.
  - `feature_analysis/`: Feature importance analysis.
    - `analyze_feature_importance_shap.py`: SHAP-based feature importance analysis.
  - `modeling/`: DR discovery and activity prediction.
    - `DR_prediction/`: DR prediction and generalization.
      - `egnn_model.py`: EGNN model for DR prediction.
      - `gpcr_graph_builder.py`: Builds GPCR binding site graphs.
      - `predict_drs_with_egnn.py`: Predicts DRs and activity with EGNN.
    - `extract_dr.py`: Extracts DRs by type.
    - `parse_gpcr_structures.py`: Analyzes GPCR structures (e.g., PDB chain selection, ligand extraction).
    - `train_model.py`: Trains the activity prediction model from scratch using pre-computed features.
    - `predict_gpcr_activity.py`: Predicts GPCR activity for new data using the pre-trained model.
  - `utils/`: Utility scripts.
    - `utils_structure.py`: Additional structure parsing utilities.
    - `split_gpcr_dataset.py`: Splits dataset into train/test sets.
- **`outputs/`**: Results and figures (to be populated post-execution).


## Installation
To set up the environment and run the code, follow these steps:
1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/hyojin0912/HJ-DR-GPCRact.git](https://github.com/hyojin0912/HJ-DR-GPCRact.git)
    cd HJ-DR-GPCRact
    ```
2.  **Install dependencies**:
    We recommend using `conda` to manage your environment for reproducibility.
    
    **Using `environment.yml` (Recommended)**:
    ```bash
    conda env create -f environment.yml
    conda activate HJ-DR-GPCRact
    ```
    **Using `pip`**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This section outlines the general workflow for reproducing the results presented in our study.

### 1. Data Preparation
First, prepare the dataset splits:
```bash
python scripts/data_preparation/split_gpcr_dataset.py
```
### 2. Modeling
Proceed with structure analysis, DR extraction, and activity prediction:
- Structure Analysis:
```bash
python scripts/modeling/parse_gpcr_structures.py
```
- DR Extraction:
```bash
python scripts/modeling/extract_dr.py
```
- GPCR Activity Prediction (DR-based models):
```bash
python scripts/modeling/predict_gpcr_activity.py
```
- DR Prediction with EGNN (for generalizability):
```bash
python scripts/modeling/DR_prediction/predict_drs_with_egnn.py
```
### 3. Feature Analysis
To analyze feature importance using SHAP:
```bash
python scripts/feature_analysis/analyze_feature_importance_shap.py
```
All generated output files, including models and results, will be saved in the `outputs/` directory.


## Figures
This section showcases key figures generated from our analysis, illustrating the biological relevance of Differential Residues and our overall workflow.

* **Figure 1: Overall Workflow and Differential Residue Visualization**
    * **(a) Schematic workflow for predicting Differential Residues (DRs) and mechanism of action (MoA) in GPCRs with unknown MoA annotations.** This panel outlines the data sources, EGNN-based DR prediction process, DR-guided MoA prediction, and structural alignment for validation.
    * **(b) MLNR (Erythromycin-bound) with Differential Residues.** This structural overlay highlights DRs (Phe173, Leu341, Phe314) in MLNR upon erythromycin binding, showing conformational changes between the Holo (8IBU) and Apo (8IBV) states.
    * **(c) HTR2B (Methylergonovine-bound) with Differential Residues.** This structural overlay emphasizes DRs (Asp135, Phe340, Phe341) in HTR2B upon methylergonovine binding, illustrating conformational changes between the Holo (6DRY) and Apo (AF2) states.
![Figure6](https://github.com/user-attachments/assets/f68fd38f-22e6-49dc-a0e7-61a617889b3a)


## Contact
For any inquiries regarding this work, please contact:
- Hyojin Son (hyojin0912@kaist.ac.kr)
- Prof. Gwan-Su Yi (gwansuyi@kaist.ac.kr)

## License
The code in this repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

The data in the `data` folder is licensed under the CC BY 4.0 International license. See the [LICENSE](./LICENSE) file for more details.
