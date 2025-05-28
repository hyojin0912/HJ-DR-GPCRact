#!/usr/bin/env python
# coding: utf-8

"""
predict_drs_with_egnn.py: Predict Differential Residues (DRs) and their coordinates for GPCRs using a pretrained EGNN model.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_undirected
from sklearn.metrics import balanced_accuracy_score
from parse_gpcr_structures import *
from gpcr_graph_builder import *
from egnn_model import *
import esm
from tqdm import tqdm

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Global settings
MODEL_PATH = "../Output/Final/EGNN_DR_Pred/final_model.pt"
GRAPH_DIR = "/home/hyojin0912/Activity/Data/GPCR_BR_Graph/"
OUTPUT_DIR = "../Output/Final/EGNN_DR_Pred/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def centroid_stats(coords: np.ndarray) -> tuple:
    """Compute centroid, mean distance, and std distance for coordinates."""
    if coords.size == 0:
        return np.zeros(3, np.float32), 0.0, 0.0
    ctr = coords.mean(0).astype(np.float32)
    if len(coords) <= 1:
        return ctr, 0.0, 0.0
    d = np.linalg.norm(coords[:, None] - coords, axis=-1)
    iu = np.triu_indices(len(coords), 1)
    dist = d[iu]
    return ctr, float(dist.mean()), float(dist.std())

def load_graph(uniprot_id: str) -> tuple:
    """Load precomputed graph or create a new one."""
    graph_path = os.path.join(GRAPH_DIR, f"{uniprot_id}.pt")
    if os.path.exists(graph_path):
        try:
            graph = torch.load(graph_path, map_location="cpu")
            graph.uniprot_id = uniprot_id
            return graph, True
        except Exception as e:
            print(f"[ERR] Failed to load graph for {uniprot_id}: {e}")
            return None, False

    bs_df = pd.read_csv("../Output/Binding_Residue/GPCR_Binding_Residues_UniProt.csv")
    rep_apo_df = pd.read_csv("../Output/Final/Representative_Apo_Structures.csv", dtype={'PDB_ID': str})
    rep_chain_df = pd.read_csv("../Output/Binding_Residue/Rep_GPCR_chain.csv", dtype={'PDB_ID': str})

    row = bs_df[bs_df['UniProt_ID'] == uniprot_id]
    if row.empty:
        print(f"[ERR] No binding residues for {uniprot_id}")
        return None, False

    binding_residues = eval(row['Binding_Residues'].iloc[0])
    apo_row = rep_apo_df[rep_apo_df['UniProt_ID'] == uniprot_id]
    if not apo_row.empty:
        apo_pdb = apo_row['PDB_ID'].iloc[0]
        apo_type = 'PDB'
        sub = rep_chain_df[(rep_chain_df['UniProt_ID'] == uniprot_id) & (rep_chain_df['PDB_ID'] == apo_pdb)]
        apo_chain = 'A' if sub.empty else sub.loc[sub['score'].astype(float).idxmax(), 'chain_id']
    else:
        apo_type = 'AF2'
        apo_pdb = 'AF2'
        apo_chain = 'A'

    graph, success = create_graph(uniprot_id, binding_residues, apo_pdb, apo_type, apo_chain)
    if success:
        torch.save(graph, graph_path)
        print(f"[INFO] Saved new graph for {uniprot_id}")
    return graph, success

def predict_drs(model, loader, device, threshold=0.5) -> tuple:
    """Predict DR probabilities and coordinates."""
    model.eval()
    results = []
    coord_dict = {}
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            uniprot_id = data.uniprot_id[0] if isinstance(data.uniprot_id, list) else data.uniprot_id
            logits, coords = model(data.x, data.pos, data.edge_index)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            res_nums = data.residue_numbers.cpu().numpy()
            preds = (probs >= threshold).astype(int)

            for i, (res_num, prob, pred) in enumerate(zip(res_nums, probs, preds)):
                results.append({
                    'UniProt_ID': uniprot_id,
                    'Residue_Number': int(res_num),
                    'Prob_DR': float(prob),
                    'Pred_DR': int(pred)
                })

            dr_indices = np.where(preds == 1)[0]
            dr_list = res_nums[dr_indices].tolist()
            dr_coords = coords[dr_indices].cpu() if dr_indices.size > 0 else None
            coord_dict[uniprot_id] = {'pred_drs': dr_list, 'pred_dr_coords': dr_coords}

    return pd.DataFrame(results), coord_dict

def optimize_threshold(df: pd.DataFrame) -> tuple:
    """Optimize per-GPCR threshold for DR prediction."""
    dr_residue_dict = {}
    thresholds = np.linspace(0.1, 0.9, 81)
    for uniprot_id, group in df.groupby('UniProt_ID'):
        probs = group['Prob_DR'].values
        preds = group['Pred_DR'].values
        if len(np.unique(preds)) < 2:
            continue
        best_thresh = 0.5
        best_bacc = 0.0
        for t in thresholds:
            new_preds = (probs >= t).astype(int)
            bacc = balanced_accuracy_score(preds, new_preds)
            if bacc > best_bacc:
                best_bacc = bacc
                best_thresh = t
        best_residues = group.loc[group['Prob_DR'] >= best_thresh, 'Residue_Number'].tolist()
        if best_residues:
            dr_residue_dict[uniprot_id] = {'threshold': best_thresh, 'residues': best_residues, 'bacc': best_bacc}
    residue_list_df = pd.DataFrame([
        {'UniProt_ID': k, 'Residue_List': '|'.join(map(str, v['residues']))}
        for k, v in dr_residue_dict.items()
    ])
    return residue_list_df, dr_residue_dict

def generate_dr1286(uniprot_id: str, dr_list: list, dr_coords: torch.Tensor, seq: str) -> np.ndarray:
    """Generate 1286-dim DR feature vector."""
    esm1280 = esm_vec(seq, dr_list)
    coords_arr = dr_coords.numpy() if dr_coords is not None else np.empty((0, 3), dtype=np.float32)
    centroid, mean_dist, std_dist = centroid_stats(coords_arr)
    return np.concatenate([
        esm1280, centroid, np.array([mean_dist, std_dist, len(dr_list)], np.float32)
    ]).astype(np.float32)

def predict_drs_with_egnn(input_df: pd.DataFrame, debug: bool = False) -> tuple:
    """Predict DRs and coordinates for GPCR binding residues."""
    model = EGNN(
        in_node_nf=1280, hidden_nf=128, out_node_nf=1, n_layers=4, residual=True, attention=True,
        normalize=True, coords_agg='mean', tanh=True, device=device, dropout=0.2
    )
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
    except Exception as e:
        print(f"[ERR] Failed to load model: {e}")
        return None, None

    graphs = []
    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Loading graphs"):
        uniprot_id = row['UniProt_ID']
        graph, success = load_graph(uniprot_id)
        if success:
            graphs.append(graph)
        elif debug:
            print(f"[DEBUG][{uniprot_id}] Failed to load or create graph")

    if not graphs:
        print("[ERR] No valid graphs generated")
        return None, None

    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    pred_df, coord_dict = predict_drs(model, loader, device)
    if pred_df.empty:
        print("[ERR] No DR predictions generated")
        return None, None

    residue_list_df, dr_residue_dict = optimize_threshold(pred_df)
    dr_features = []
    for uniprot_id in pred_df['UniProt_ID'].unique():
        rep_apo_df = pd.read_csv("../Output/Final/Representative_Apo_Structures.csv", dtype={'PDB_ID': str})
        seq = get_sequence_from_structure(
            load_structure_mmCIF(rep_apo_df[rep_apo_df['UniProt_ID'] == uniprot_id]['PDB_ID'].iloc[0]) if uniprot_id in rep_apo_df['UniProt_ID'].values
            else load_structure_AF2(uniprot_id),
            'A'
        )
        if uniprot_id in dr_residue_dict:
            dr_list = dr_residue_dict[uniprot_id]['residues']
            dr_coords = coord_dict.get(uniprot_id, {}).get('pred_dr_coords')
            dr_vec = generate_dr1286(uniprot_id, dr_list, dr_coords, seq)
            dr_features.append({'AC': uniprot_id, 'DR1286': dr_vec.tolist()})

    dr_feature_df = pd.DataFrame(dr_features)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "dr_predictions.csv"), index=False)
    residue_list_df.to_csv(os.path.join(OUTPUT_DIR, "dr_residue_list.csv"), index=False)
    dr_feature_df.to_csv(os.path.join(OUTPUT_DIR, "dr_features.csv"), index=False)
    torch.save(coord_dict, os.path.join(OUTPUT_DIR, "dr_coords.pt"))
    print(f"[INFO] Saved predictions to {OUTPUT_DIR}")

    return pred_df, coord_dict

def debug_prediction(uniprot_id: str, binding_residues: list) -> tuple:
    """Debug DR prediction for a single GPCR."""
    print(f"[DEBUG] Predicting DRs for UniProt_id: {uniprot_id}")
    input_df = pd.DataFrame([{'uniprot_id': uniprot_id, 'Binding_Residues': str(binding_residues)}])
    return predict_drs_with_egnn(input_df, debug=True)

if __name__ == "__main__":
    input_csv = "../Input/dr_prediction_input.csv"
    if os.path.exists(input_csv):
        input_df = pd.read_csv(input_csv)
        predict_drs_with_egnn(input_df)
    else:
        print(f"[ERR] Input CSV not found at {input_csv}")
        debug_prediction(uniprot_id="P08913", binding_residues=[123, 456, 789])