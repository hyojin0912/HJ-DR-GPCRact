# HJ-DR-GPCRact

## Overview
This repository contains code and data for the study *"Interpretable prediction of ligand-induced GPCR activity via structure-based differential residue modeling"*, submitted to the *Nature Communications*. The work introduces GPCRactDB, a curated dataset of 202,925 ligand-GPCR interactions, and proposes Differential Residues (DRs)—residues with ligand-induced conformational changes—as interpretable features for predicting GPCR activity.

### Key Findings
- **GPCR-specific DR patterns**: DRs reflect structural variability across GPCRs; some remain rigid while others exhibit strong dynamic shifts.
- **High DR consistency**: DRs are conserved across ligands for the same GPCR (consistency score > 0.8).
- **Robust interpretable model**: Outperforms sequence-based deep learning models ([DeepREAL](https://academic.oup.com/bioinformatics/article/38/9/2561/6547052), [AiGPRo](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00945-7)) under scaffold-based splits.
- **Generalizability**: Applicable to data-scarce GPCRs using EGNN-predicted DRs (AUROC > 0.8).
- **Experimental relevance**: DRs align with biologically validated residues in HCAR2, MLNR, and HTR2B.

## Repository Structure
- **`data/`**: Contains GPCR-related data.
  - `Human_GPCR_PDB_Info.csv`: CSV file with UniProt and PDB information.
  - `splits/`: Train and test sets from pair and scaffold splits (InChIKey | UniProt accession | Label).
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
    - `predict_gpcr_activity.py`: Builds and predicts with DR-based models.
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

## Figures
This section showcases key figures generated from our analysis, illustrating the biological relevance of Differential Residues.

Figure 1: MLNR (Erythromycin-bound): Shows DRs (Phe173, Leu341, Phe314) in MLNR.
Figure 2: HTR2B (Methylergonovine-bound): Highlights DRs (Asp135, Phe340, Phe341) in HTR2B.
![image](https://github.com/user-attachments/assets/42ab74a1-69a7-49ec-8747-967bdb5e664e)

## Contact
