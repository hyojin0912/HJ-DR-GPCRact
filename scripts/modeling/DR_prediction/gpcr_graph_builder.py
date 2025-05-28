#!/usr/bin/env python
# coding: utf-8

"""
gpcr_graph_builder.py: Create graph representations for GPCR binding residues.
Handles structure loading, sequence alignment, ESM-1b embeddings, and graph generation.
"""

import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
import esm
from Bio.PDB import MMCIFParser, PDBParser
import Bio.PDB.Polypeptide
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from tqdm import tqdm
from parse_gpcr_structures import *

# Constants
BS_FILE = "../Output/Binding_Residue/GPCR_Binding_Residues_UniProt.csv"
REPR_APO_CSV = "../Output/Final/Representative_Apo_Structures.csv"
REP_CHAIN_CSV = "../Output/Binding_Residue/Rep_GPCR_chain.csv"
CIF_DIR = "../Data/CIF_Files"
AF_DIR = "../Data/AF_PDB"
GRAPH_OUTPUT_DIR = "/home/hyojin0912/Activity/Data/GPCR_BR_Graph"
EDGE_CUTOFF = 8.0  # Distance cutoff for edges (Å)
BLOSUM62 = matlist.blosum62
GAP_OPEN_PENALTY = -10
GAP_EXTEND_PENALTY = -0.5

# Ensure output directory exists
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Maximum number of tokens including special BOS/EOS
MAX_TOKENS = model.embed_positions.max_positions  # Typically 1024

@torch.no_grad()
def esm_vec(seq: str, idx_list: list) -> np.ndarray:
    """
    Compute mean-pooled 1280-d ESM vectors at given 1-based indices.
    Handles long sequences by slicing subsequences around each index.
    """
    usable_len = MAX_TOKENS - 2  # Account for BOS/EOS tokens
    vectors = []
    for idx in idx_list:
        pos0 = idx - 1  # Convert to 0-based
        half = usable_len // 2
        start = max(0, pos0 - half)
        end = min(len(seq), start + usable_len)
        if end - start < usable_len:
            start = max(0, end - usable_len)
        subseq = seq[start:end]
        _, _, toks = batch_converter([("p", subseq)])
        toks = toks.to(device)
        rep = model(toks, repr_layers=[33])["representations"][33][0, 1:-1]
        rel_idx = pos0 - start
        if 0 <= rel_idx < rep.size(0):
            vectors.append(rep[rel_idx].cpu().numpy().astype(np.float32))
    return np.mean(vectors, axis=0).astype(np.float32) if vectors else np.zeros(1280, dtype=np.float32)

def load_structure_mmCIF(pdb_id: str, cif_dir: str = CIF_DIR) -> object:
    """Load mmCIF structure."""
    cif_path = os.path.join(cif_dir, f"{pdb_id.lower()}.cif")
    if not os.path.exists(cif_path):
        print(f"[WARNING] mmCIF file not found for {pdb_id} at {cif_path}")
        return None
    parser = MMCIFParser(QUIET=True)
    try:
        return parser.get_structure(pdb_id, cif_path)
    except Exception as e:
        print(f"[ERROR] Parsing mmCIF for {pdb_id}: {e}")
        return None

def load_structure_AF2(uniprot_id: str, af_dir: str = AF_DIR) -> object:
    """Load AlphaFold2 PDB file."""
    af_file = os.path.join(af_dir, f"AF-{uniprot_id}-F1-model_v3.pdb")
    if not os.path.exists(af_file):
        print(f"[WARNING] AF2 PDB file not found at {af_file}")
        return None
    parser = PDBParser(QUIET=True)
    try:
        return parser.get_structure(uniprot_id, af_file)
    except Exception as e:
        print(f"[ERROR] Parsing AF2 PDB file {af_file}: {e}")
        return None

def get_sequence_from_structure(structure: object, chain_id: str) -> str:
    """Extract sequence from the specified chain."""
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                ppb = Bio.PDB.Polypeptide.PPBuilder()
                sequence = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(chain))
                return sequence
    return ""

def get_ca_coordinates(structure: object, chain_id: str, binding_residues: list, res_map: dict) -> tuple:
    """Extract Cα coordinates for UniProt residue numbers."""
    coords = []
    matched_residues = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue in res_map and res_map[residue] in binding_residues and 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
                        matched_residues.append(res_map[residue])
    return np.array(coords, dtype=np.float32), matched_residues

def create_graph(uniprot_id: str, binding_residues: list, apo_pdb: str, apo_type: str, chain_id: str) -> tuple:
    """
    Create a graph for a GPCR with binding residues as nodes.
    Returns (graph, success_flag).
    """
    print(f"[INFO] Processing {uniprot_id} with {apo_type} (PDB: {apo_pdb}, Chain: {chain_id})")
    structure = load_structure_mmCIF(apo_pdb) if apo_type == 'PDB' else load_structure_AF2(uniprot_id)
    if structure is None:
        print(f"[SKIP] Failed to load structure for {uniprot_id} ({apo_type})")
        return None, False

    sequence = get_sequence_from_structure(structure, chain_id)
    if not sequence:
        print(f"[SKIP] Failed to extract sequence for {uniprot_id} (chain {chain_id})")
        return None, False

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                pdb_res_list = [res for res in chain if Bio.PDB.Polypeptide.is_aa(res, standard=True)]
                break
        else:
            continue
        break
    else:
        print(f"[SKIP] Chain {chain_id} not found in structure for {uniprot_id}")
        return None, False

    aln_res = local_align_smith_waterman(sequence, sequence)
    if not aln_res["aligned_pdb_seq"]:
        print(f"[SKIP] Alignment failed for {uniprot_id} (chain {chain_id})")
        return None, False

    res_map = map_pdb_res_to_uniprot(pdb_res_list, aln_res["aligned_pdb_seq"], aln_res["aligned_uni_seq"])
    if not res_map:
        print(f"[SKIP] Residue mapping failed for {uniprot_id} (chain {chain_id})")
        return None, False

    coords, matched_residues = get_ca_coordinates(structure, chain_id, binding_residues, res_map)
    if len(matched_residues) != len(binding_residues):
        print(f"[SKIP] Mismatch in coordinates for {uniprot_id}: expected {len(binding_residues)}, got {len(matched_residues)}")
        return None, False
    if len(coords) == 0:
        print(f"[SKIP] No coordinates extracted for {uniprot_id}")
        return None, False

    node_features = torch.tensor([esm_vec(sequence, [res_num]) for res_num in matched_residues], dtype=torch.float)
    pos = torch.tensor(coords, dtype=torch.float)

    edge_index = []
    for i in range(len(matched_residues)):
        for j in range(i + 1, len(matched_residues)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < EDGE_CUTOFF:
                edge_index.extend([[i, j], [j, i]])
    if not edge_index:
        print(f"[SKIP] No edges created for {uniprot_id} (num_nodes: {len(matched_residues)})")
        return None, False

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    graph = Data(
        x=node_features,
        pos=pos,
        edge_index=edge_index,
        uniprot_id=uniprot_id,
        residue_numbers=torch.tensor(matched_residues, dtype=torch.long)
    )
    return graph, True

def main():
    """Generate and save graphs for all GPCRs in BS_FILE."""
    bs_df = pd.read_csv(BS_FILE)
    rep_apo_df = pd.read_csv(REPR_APO_CSV, dtype={'PDB_ID': str})
    rep_chain_df = pd.read_csv(REP_CHAIN_CSV, dtype={'PDB_ID': str})

    graphs = []
    failed_gpcrs = []
    apo_attempted = 0

    for idx, row in tqdm(bs_df.iterrows(), total=len(bs_df), desc="Processing GPCRs"):
        uniprot_id = row['UniProt_ID']
        binding_residues = eval(row['Binding_Residues'])
        apo_row = rep_apo_df[rep_apo_df['UniProt_ID'] == uniprot_id]
        if not apo_row.empty:
            apo_pdb = apo_row['PDB_ID'].iloc[0]
            apo_type = 'PDB'
            sub = rep_chain_df[(rep_chain_df['UniProt_ID'] == uniprot_id) & (rep_chain_df['PDB_ID'] == apo_pdb)]
            apo_chain = 'A' if sub.empty else sub.loc[sub['score'].astype(float).idxmax(), 'chain_id']
            apo_attempted += 1
        else:
            apo_type = 'AF2'
            apo_pdb = 'AF2'
            apo_chain = 'A'

        try:
            graph, success = create_graph(uniprot_id, binding_residues, apo_pdb, apo_type, apo_chain)
            if success:
                graphs.append(graph)
                torch.save(graph, os.path.join(GRAPH_OUTPUT_DIR, f"{uniprot_id}.pt"))
                print(f"[SAVE] Graph for {uniprot_id}: {graph.x.size(0)} nodes, {graph.edge_index.size(1)} edges using {apo_type}")
            else:
                failed_gpcrs.append(uniprot_id)
        except Exception as e:
            print(f"[ERROR] Failed to create graph for {uniprot_id} with {apo_type}: {e}")
            failed_gpcrs.append(uniprot_id)

    print(f"[INFO] Attempted Apo PDB for {apo_attempted} GPCRs, failed for {len(failed_gpcrs)} GPCRs")
    failed_with_af2 = []
    for uniprot_id in failed_gpcrs:
        row = bs_df[bs_df['UniProt_ID'] == uniprot_id]
        binding_residues = eval(row['Binding_Residues'].iloc[0])
        try:
            graph, success = create_graph(uniprot_id, binding_residues, 'AF2', 'AF2', 'A')
            if success:
                graphs.append(graph)
                torch.save(graph, os.path.join(GRAPH_OUTPUT_DIR, f"{uniprot_id}.pt"))
                print(f"[SAVE] Graph for {uniprot_id}: {graph.x.size(0)} nodes, {graph.edge_index.size(1)} edges using AF2")
            else:
                failed_with_af2.append(uniprot_id)
        except Exception as e:
            print(f"[ERROR] Failed to create graph for {uniprot_id} with AF2: {e}")
            failed_with_af2.append(uniprot_id)

    graphs_dict = {graph.uniprot_id: graph for graph in graphs}
    graphs = list(graphs_dict.values())
    graphs = [graph for graph in graphs if graph.uniprot_id not in failed_with_af2]
    print(f"[INFO] Generated and saved {len(graphs)} graphs.")
    print(f"[INFO] Failed with Apo PDB or no Apo PDB: {failed_gpcrs}")
    print(f"[INFO] Failed with AF2: {failed_with_af2}")

if __name__ == "__main__":
    main()