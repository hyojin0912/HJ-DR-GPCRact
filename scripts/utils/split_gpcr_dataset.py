#!/usr/bin/env python
# coding: utf-8

"""
split_gpcr_dataset.py: Perform scaffold-based and pair-based splitting of GPCR ligand data
for drug activity prediction. Generates train/test/dev splits with chemical diversity and class balance.
"""

import pandas as pd
import numpy as np
import random
import os
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from collections import defaultdict
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Constants
TEST_FRAC = 0.25  # Target fraction for test set
DEV_FRAC = 0.2    # Fraction of train set for dev set (for DeepREAL)
FP_RADIUS = 2     # Morgan fingerprint radius
FP_NBITS = 1024   # Morgan fingerprint bits
CLUST_THRESH = 0.35  # Tanimoto threshold for Butina clustering
LABEL_TOL = 0.1   # Allowed deviation in class ratio
DISP_THRESHOLD_AGO = 2.0  # Å threshold for DR-Ago
DISP_THRESHOLD_ANT = 2.0  # Å threshold for DR-Ant
DISP_THRESHOLD_STATE = 0.5  # Å threshold for DR-State
OUTPUT_DIR = "../Output/Final/Dataset_Splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_filter_data(assay_csv: str, pdb_csv: str, dr_csv: str) -> pd.DataFrame:
    """Load and filter ligand data, merging assay and PDB datasets."""
    logger.info("Loading and filtering data")
    
    # Load assay data
    assay_df = pd.read_csv(assay_csv)
    assay_df = assay_df[['Ikey', 'AC', 'Label', 'SMILES']]
    assay_df = assay_df[assay_df['Label'].isin(['agonist', 'antagonist'])]

    # Load PDB data
    pdb_df = pd.read_csv(pdb_csv)
    pdb_df = pdb_df[['InChIKey', 'Entry', 'MoA', 'SMILES']]
    pdb_df.columns = ['Ikey', 'AC', 'Label', 'SMILES']

    # Concatenate and remove duplicates
    data = pd.concat([assay_df, pdb_df], ignore_index=True).drop_duplicates()

    # Load DR data and filter for non-empty DR-State
    dr_df = pd.read_csv(dr_csv)
    dr_df = dr_df[dr_df['DR-State'].astype(str) != '[]'].reset_index(drop=True)
    valid_uniprots = set(dr_df['UniProt_ID'])

    # Filter data by valid UniProt IDs
    data = data[data['AC'].isin(valid_uniprots)].reset_index(drop=True)

    # Validate SMILES
    data['Mol'] = data['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
    invalid_smiles = data['Mol'].isna()
    if invalid_smiles.any():
        logger.warning(f"Removing {invalid_smiles.sum()} invalid SMILES")
        data = data[~invalid_smiles].reset_index(drop=True)

    logger.info(f"Loaded {len(data)} valid compounds")
    return data.drop(columns=['Mol'])

def compute_scaffold_and_fp(smiles: str) -> tuple:
    """Compute Murcko scaffold and Morgan fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    scaf = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
    scaf_smiles = Chem.MolToSmiles(scaf, isomericSmiles=False)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
    return scaf_smiles, fp

def butina_cluster(fp_list: list, threshold: float = CLUST_THRESH) -> list:
    """Perform Butina clustering on fingerprints."""
    logger.info("Performing Butina clustering")
    n = len(fp_list)
    dists = []
    for i in tqdm(range(1, n), desc="Computing distances"):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, nPts=n, distThresh=1 - threshold, isDistData=True)
    return clusters

def scaffold_split(data: pd.DataFrame) -> pd.DataFrame:
    """Perform scaffold-based train/test split."""
    logger.info("Computing scaffolds and fingerprints")
    
    # Compute scaffolds and fingerprints
    scaffolds, fps = [], []
    for smiles in tqdm(data['SMILES'], desc="Processing SMILES"):
        scaf, fp = compute_scaffold_and_fp(smiles)
        scaffolds.append(scaf)
        fps.append(fp)
    
    data['Scaffold'] = scaffolds
    data['FP'] = fps
    data = data.dropna(subset=['Scaffold', 'FP']).reset_index(drop=True)

    # Perform clustering
    clusters = butina_cluster(data['FP'].tolist())
    
    # Map indices to cluster IDs
    cluster_id = {idx: cid for cid, cluster in enumerate(clusters) for idx in cluster}
    data['Cluster'] = data.index.map(cluster_id)

    # Stratified sampling
    cluster_indices = defaultdict(list)
    for idx, cid in cluster_id.items():
        cluster_indices[cid].append(idx)

    all_cids = list(cluster_indices.keys())
    random.shuffle(all_cids)

    target_test = int(len(data) * TEST_FRAC)
    test_idx = set()
    label_counts = data['Label'].value_counts().to_dict()

    def ratio_ok(test_ids):
        test_labels = data.loc[list(test_ids), 'Label'].value_counts()
        ratio = (test_labels / len(test_ids)).reindex(label_counts.index).fillna(0)
        base = pd.Series(label_counts) / len(data)
        return ((ratio - base).abs() <= LABEL_TOL).all()

    logger.info("Sampling clusters for test set")
    for cid in tqdm(all_cids, desc="Sampling clusters"):
        cand = set(cluster_indices[cid])
        if len(test_idx) + len(cand) > target_test * 1.05:
            continue
        if ratio_ok(test_idx | cand):
            test_idx.update(cand)
        if len(test_idx) >= target_test:
            break

    data['split'] = 'train'
    data.loc[list(test_idx), 'split'] = 'test'

    # Compute inter-set similarity
    train_fps = [data.loc[i, 'FP'] for i in data.index if data.loc[i, 'split'] == 'train']
    test_fps = [data.loc[i, 'FP'] for i in test_idx]

    def max_cross_similarity(tfps, tfps2):
        for fp in tfps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, tfps2)
            yield max(sims)

    similarities = list(max_cross_similarity(train_fps, test_fps))
    logger.info(f"Max train-test Tanimoto: {max(similarities):.3f}")
    logger.info(f"Mean train-test Tanimoto: {np.mean(similarities):.3f}")
    logger.info(f"Min train-test Tanimoto: {min(similarities):.3f}")

    return data

def pair_split(data: pd.DataFrame) -> tuple:
    """Perform random pair-based train/test split."""
    logger.info("Performing pair-based split")
    train_data, test_data = train_test_split(
        data,
        test_size=TEST_FRAC,
        stratify=data['Label'],
        random_state=SEED
    )
    logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    return train_data, test_data

def main():
    """Main function to split GPCR dataset."""
    # Input files
    assay_csv = '../Input_Data/data_wanted.csv'
    pdb_csv = '../Input_Data/Binding/DR_info_PDB.csv'
    dr_csv = f"../Output/Final_Data/AF2/GPCR_DR_sets_{DISP_THRESHOLD_AGO}_{DISP_THRESHOLD_ANT}_{DISP_THRESHOLD_STATE}.txt"

    # Load and filter data
    data = load_and_filter_data(assay_csv, dr_csv)

    # --- Pair-based Split ---
    logger.info("Generating pair-based split")
    train_pair, test_pair = pair_split(data.copy())
    train_pair[['Ikey', 'AC', 'Label', 'SMILES']].to_csv(f"{OUTPUT_DIR}/train_set_pair.csv", index=False)
    test_pair[['Ikey', 'AC', 'Label', 'SMILES']].to_csv(f"{OUTPUT_DIR}/test_set_pair.csv", index=False)

    # Prepare DeepREAL splits for pair-based
    pair_deepreal_dir = os.path.join(DEEPREAL_DIR, "Final_pair")
    os.makedirs(pair_deepreal_dir, exist_ok=True)
    prepare_deepreal_splits(train_pair.copy(), test_pair.copy(), pair_deepreal_dir)

    # --- Scaffold-based Split ---
    logger.info("Generating scaffold-based split")
    scaffold_data = scaffold_split(data.copy())
    
    # Save scaffold-based splits
    cols = ['Ikey', 'AC', 'Label', 'SMILES']
    scaffold_data[scaffold_data['split'] == 'train'][cols].to_csv(f"{OUTPUT_DIR}/train_set.csv", index=False)
    scaffold_data[scaffold_data['split'] == 'test'][cols].to_csv(f"{OUTPUT_DIR}/test_set.csv", index=False)

    logger.info("Dataset splitting completed.")

if __name__ == "__main__":
    main()