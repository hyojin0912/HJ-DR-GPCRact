#!/usr/bin/env python
# coding: utf-8

"""
predict_gpcr_activity.py: Predict GPCR activity (agonist vs. antagonist) for ligand-GPCR pairs using a pretrained MLP model.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import esm
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import combinations
from tqdm import tqdm
from parse_gpcr_structures import *
from extract_dr import extract_dr_ago, extract_dr_ant, extract_dr_state
from tensorflow.keras.models import load_model

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# --- Global Settings ---
# NOTE: Update these paths to match your repository structure
MODEL_PATH = "../Output/Final/AF2/model"  # Path to the saved Keras model directory or .h5 file

# DR thresholds (ensure these match the training configuration)
DISP_THRESHOLD_AGO = 2.0
DISP_THRESHOLD_ANT = 2.0
DISP_THRESHOLD_STATE = 0.5

DR_COORD_PATH = f"../Data/Feature/AF2/Final_GPCR_all_DR_labels_{DISP_THRESHOLD_AGO}_{DISP_THRESHOLD_ANT}_{DISP_THRESHOLD_STATE}.pt"
DR_SETS_PATH = f"../Output/Final/AF2/GPCR_DR_sets_thr_{DISP_THRESHOLD_AGO}_{DISP_THRESHOLD_ANT}_{DISP_THRESHOLD_STATE}.csv"
OUTPUT_DIR = "../Output/Final/AF2/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ESM-1b model once
print("[INFO] Loading ESM-1b model...")
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model = esm_model.to(device)
print(f"[INFO] ESM-1b model loaded to {device}.")

@torch.no_grad()
def esm_vec(seq, idx_list):
    """Return mean-pooled 1280-dim ESM vector over given UniProt indices."""
    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    representations = esm_model(tokens, repr_layers=[33])["representations"][33][0, 1:-1]  # [L, 1280]
    vectors = [representations[i-1].cpu().numpy() for i in idx_list if 0 < i <= representations.size(0)]
    return np.mean(vectors, axis=0).astype(np.float32) if vectors else np.zeros(1280, np.float32)

def centroid_stats(coords):
    """Compute centroid, mean distance, and std distance from coordinates [N, 3]."""
    if coords.size == 0:
        return np.zeros(3, np.float32), 0.0, 0.0
    centroid = coords.mean(0).astype(np.float32)
    if len(coords) == 1:
        return centroid, 0.0, 0.0
    distances = np.linalg.norm(coords[:, None] - coords, axis=-1)
    iu = np.triu_indices(len(coords), 1)
    dist_values = distances[iu]
    return centroid, float(dist_values.mean()), float(dist_values.std())

def generate_ecfp4(smiles):
    """Generate 1024-bit ECFP4 fingerprint from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(1024, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        return np.array(fp, dtype=np.float32)
    except Exception as e:
        print(f"[ERR] generate_ecfp4: {e}")
        return np.zeros(1024, dtype=np.float32)

def generate_dr1286(uniprot_id, dr_sets_df, dr_coord_dict):
    """Generate 1286-dim DR feature vector (1280 ESM-1b + 6 geometric)."""
    dr_row = dr_sets_df[dr_sets_df['UniProt_ID'] == uniprot_id]
    if dr_row.empty:
        print(f"[WARN] No DR data for {uniprot_id}")
        return np.zeros(1286, dtype=np.float32)

    # Parse DR lists
    dr_ago = set(ast.literal_eval(dr_row['DR-Ago'].iloc[0]))
    dr_ant = set(ast.literal_eval(dr_row['DR-Ant'].iloc[0]))
    dr_state = set(ast.literal_eval(dr_row['DR-State'].iloc[0]))
    dr_all = sorted(dr_ago | dr_ant | dr_state)
    if not dr_all:
        print(f"[WARN] Empty DR set for {uniprot_id}")
        return np.zeros(1286, dtype=np.float32)

    # Fetch ESM-1b features
    seq = fetch_uniprot_sequence(uniprot_id)
    if not seq:
        print(f"[ERR] No sequence for {uniprot_id}")
        return np.zeros(1286, dtype=np.float32)
    esm1280 = esm_vec(seq, dr_all)

    # Extract coordinates
    coords_list = []
    if uniprot_id in dr_coord_dict:
        entry = dr_coord_dict[uniprot_id]
        def pick_coords(c_arr, r_list, keep_set):
            if c_arr is None or not r_list:
                return []
            c_np = c_arr.cpu().numpy() if isinstance(c_arr, torch.Tensor) else np.asarray(c_arr)
            return [c_np[i] for i, rid in enumerate(r_list) if rid in keep_set]

        coords_list += pick_coords(entry.get("agonist_dr_coords"), entry.get("agonist_drs", []), dr_ago)
        coords_list += pick_coords(entry.get("antagonist_dr_coords"), entry.get("antagonist_drs", []), dr_ant)
        coords_list += pick_coords(entry.get("state_dr_coords"), entry.get("state_drs", []), dr_state)

    coords_arr = np.vstack(coords_list) if coords_list else np.empty((0, 3), dtype=np.float32)
    centroid, mean_dist, std_dist = centroid_stats(coords_arr)

    # Assemble DR1286 vector
    dr_vec = np.concatenate([
        esm1280,
        centroid,
        np.array([mean_dist, std_dist, len(dr_all)], dtype=np.float32)
    ])
    return dr_vec.astype(np.float32)

def predict_activity(input_df, model_path=MODEL_PATH, debug=False):
    """Predict GPCR activity for ligand-GPCR pairs."""
    # Load model
    try:
        print(f"[INFO] Loading pretrained model from {model_path}...")
        model = load_model(model_path)
        model.summary()
    except Exception as e:
        print(f"[ERR] Failed to load model: {e}")
        return None

    # Load prerequisite data files for feature generation
    print("[INFO] Loading feature generation data...")
    try:
        dr_coord_dict = torch.load(DR_COORD_PATH, map_location="cpu")
        dr_sets_df = pd.read_csv(DR_SETS_PATH)
    except FileNotFoundError as e:
        print(f"[ERR] Prerequisite data file not found: {e}")
        return None

    # Generate features
    results = []
    for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Predicting"):
        uniprot_id = row['UniProt_ID']
        ikey = row['Ikey']
        smiles = row['SMILES']

        # Generate ECFP4
        ecfp4 = generate_ecfp4(smiles)
        if debug and np.all(ecfp4 == 0):
            print(f"[DEBUG][{uniprot_id}][{ikey}] Zero ECFP4 vector")

        # Generate DR1286
        dr1286 = generate_dr1286(uniprot_id, dr_coord_dict)
        if debug and np.all(dr1286 == 0):
            print(f"[DEBUG][{uniprot_id}][{ikey}] Zero DR1286 vector")

        # Predict
        try:
            prob = model.predict([ecfp4[None, :], dr1286[None, :]], verbose=0).ravel()[0]
            pred_label = int(prob > 0.5)
            results.append({
                'UniProt_ID': uniprot_id,
                'Ikey': ikey,
                'SMILES': smiles,
                'pred_prob': float(prob),
                'pred_label': pred_label
            })
        except Exception as e:
            if debug:
                print(f"[DEBUG][{uniprot_id}][{ikey}] Prediction failed: {e}")
            results.append({
                'UniProt_ID': uniprot_id,
                'Ikey': ikey,
                'SMILES': smiles,
                'pred_prob': np.nan,
                'pred_label': np.nan
            })

    result_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, "activity_predictions.csv")
    result_df.to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to {output_path}")
    return result_df

def debug_prediction(uniprot_id, ikey, smiles):
    """Debug prediction for a single GPCR-ligand pair."""
    print(f"[DEBUG] Predicting for UniProt_ID: {uniprot_id}, Ikey: {ikey}")
    input_df = pd.DataFrame([{
        'UniProt_ID': uniprot_id,
        'Ikey': ikey,
        'SMILES': smiles
    }])
    return predict_activity(input_df, debug=True)

if __name__ == "__main__":
    # Example usage with a CSV input
    input_csv = "../Input/prediction_input.csv"  # Expected columns: UniProt_ID, Ikey, SMILES
    if os.path.exists(input_csv):
        input_df = pd.read_csv(input_csv)
        predict_activity(input_df)
    else:
        print(f"[ERR] Input CSV not found at {input_csv}. Please provide a valid input file.")
        # Example single prediction
        debug_prediction(
            uniprot_id="P08913",
            ikey="CHEMBL123456",
            smiles="C1CCC(/N=C(/NC2CCCCC2)N2CCOCC2)CC1"
        )
