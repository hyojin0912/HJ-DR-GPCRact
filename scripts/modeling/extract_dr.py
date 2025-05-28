#!/usr/bin/env python
# coding: utf-8

"""
extract_dr.py: Extract Differential Residues (DR) for GPCRs using Apo, Agonist, and Antagonist structures.
Integrates with parse_gpcr_structures.py for structure processing and supports AlphaFold2 (AF2) structures.
Generates CSV summaries and PyTorch dictionaries for DR-Ago, DR-Ant, and DR-State.
"""

import os
import numpy as np
import pandas as pd
import torch
import ast
from parse_gpcr_structures import *

# Global settings
CIF_DIR = "../Data/CIF_Files"
AF_DIR = "../Data/AF_PDB"
OUTPUT_FEATURE_DIR = "../Data/Feature/AF2"
OUTPUT_FINAL_DIR = "../Output/Final/AF2"
os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_FINAL_DIR, exist_ok=True)

# DR thresholds
DISP_THRESHOLD_AGO = 2.0    # Å for DR-Ago
DISP_THRESHOLD_ANT = 2.0    # Å for DR-Ant
DISP_THRESHOLD_STATE = 0.5  # Å for DR-State

# Output paths
DR_OUTPUT_PATH_AGO = os.path.join(OUTPUT_FEATURE_DIR, f"Final_GPCR_ago_DR_labels_{DISP_THRESHOLD_AGO}.pt")
CSV_OUTPUT_PATH_AGO = os.path.join(OUTPUT_FINAL_DIR, f"Apo_Ago_Binding_Site_RMSD_Comparison_{DISP_THRESHOLD_AGO}.csv")
DR_OUTPUT_PATH_ANT = os.path.join(OUTPUT_FEATURE_DIR, f"Final_GPCR_ant_DR_labels_{DISP_THRESHOLD_ANT}.pt")
CSV_OUTPUT_PATH_ANT = os.path.join(OUTPUT_FINAL_DIR, f"Apo_Ant_Binding_Site_RMSD_Comparison_{DISP_THRESHOLD_ANT}.csv")
DR_OUTPUT_PATH_STATE = os.path.join(OUTPUT_FEATURE_DIR, f"Final_GPCR_state_DR_labels_{DISP_THRESHOLD_STATE}.pt")
CSV_OUTPUT_PATH_STATE = os.path.join(OUTPUT_FINAL_DIR, f"Ago_Ant_Binding_Site_RMSD_Comparison_{DISP_THRESHOLD_STATE}.csv")
DR_MERGED_PATH = os.path.join(OUTPUT_FEATURE_DIR, f"Final_GPCR_all_DR_labels_{DISP_THRESHOLD_AGO}_{DISP_THRESHOLD_ANT}_{DISP_THRESHOLD_STATE}.pt")
DR_SETS_CSV = os.path.join(OUTPUT_FINAL_DIR, f"GPCR_DR_sets_thr_{DISP_THRESHOLD_AGO}_{DISP_THRESHOLD_ANT}_{DISP_THRESHOLD_STATE}.csv")

def extract_dr_ago():
    """Extract DR-Ago by comparing Apo (or AF2) vs. Agonist-bound structures."""
    # Load input data
    rep_apo_df = pd.read_csv("../Output/Final/Representative_Apo_Structures.csv", dtype={'PDB_ID': str})
    rep_apo_df = rep_apo_df[rep_apo_df.Binding_Coverage.astype(float) >= 50]
    rep_map = dict(zip(rep_apo_df.UniProt_ID, rep_apo_df.PDB_ID))
    cls_df = pd.read_csv("../Output/Final/GPCR_PDB_classification.csv")[['UniProt_ID', 'Ago_PDB']]
    lig_info = pd.read_csv("../Input/Binding_site/PDB_ago_ant_chain_info_v2.csv", dtype={'PDB_ID': str})
    rep_chain = pd.read_csv("../Output/Binding_Residue/Rep_GPCR_chain.csv", dtype={'PDB_ID': str})

    dr_dict, summary = {}, []
    for _, row in cls_df[cls_df.Ago_PDB > 0].iterrows():
        uid = row['UniProt_ID']
        apo_source = rep_map.get(uid)
        apo_info = None
        if apo_source:
            sub = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == apo_source)]
            if not sub.empty:
                apo_chain = sub.loc[sub['score'].astype(float).idxmax(), 'chain_id']
                apo_info = process_structure(uid, pdb_id=apo_source, chain_id=apo_chain, is_apo=True)
        if apo_info is None:
            af_file = os.path.join(AF_DIR, f"AF-{uid}-F1-model_v3.pdb")
            apo_info = process_structure(uid, af_path=af_file, chain_id='A', is_apo=True)
        if apo_info is None:
            continue

        coords_acc = {}
        for _, ag in lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == 'agonist')].iterrows():
            sub2 = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == ag.PDB_ID)]
            if sub2.empty:
                continue
            ag_chain = sub2.loc[sub2['score'].astype(float).idxmax(), 'chain_id']
            ag_info = process_structure(uid, pdb_id=ag.PDB_ID, chain_id=ag_chain, ligand_id=ag.LIGAND_ID, is_apo=False)
            if ag_info is None:
                continue
            common = np.intersect1d(apo_info['uni_idxs'], ag_info['uni_idxs'])
            if len(common) == 0:
                continue
            map_apo = dict(zip(apo_info['uni_idxs'], apo_info['coords']))
            map_ag = dict(zip(ag_info['uni_idxs'], ag_info['coords']))
            rmsd, disp, transformed = compute_rmsd_and_disp(
                np.array([map_apo[u] for u in common]),
                np.array([map_ag[u] for u in common])
            )
            diffs = [int(u) for u, d in zip(common, disp) if d >= DISP_THRESHOLD_AGO]
            summary.append({
                'UniProt_ID': uid,
                'Apo_Source': apo_source or 'AF2',
                'Agonist_PDB': ag.PDB_ID,
                'Common_Residues': list(common),
                'Differential_Binding_Residues': diffs,
                'RMSD': rmsd
            })
            for u, d, coord in zip(common, disp, transformed):
                if d >= DISP_THRESHOLD_AGO:
                    coords_acc.setdefault(int(u), []).append(coord)
        if coords_acc:
            final_res = sorted(coords_acc.keys())
            avg_coords = [np.mean(np.vstack(coords_acc[u]), axis=0) for u in final_res]
            dr_dict[uid] = {
                'apo_pdb': apo_source or 'AF2',
                'agonist_drs': final_res,
                'agonist_dr_coords': torch.tensor(np.vstack(avg_coords), dtype=torch.float32)
            }

    pd.DataFrame(summary).to_csv(CSV_OUTPUT_PATH_AGO, index=False)
    torch.save(dr_dict, DR_OUTPUT_PATH_AGO)
    print(f"[INFO] DR-Ago saved to {CSV_OUTPUT_PATH_AGO} and {DR_OUTPUT_PATH_AGO}")
    return dr_dict

def extract_dr_ant():
    """Extract DR-Ant by comparing Apo (or AF2) vs. Antagonist-bound structures."""
    rep_apo_df = pd.read_csv("../Output/Final/Representative_Apo_Structures.csv", dtype={'PDB_ID': str})
    rep_apo_df = rep_apo_df[rep_apo_df.Binding_Coverage.astype(float) >= 50]
    rep_map = dict(zip(rep_apo_df.UniProt_ID, rep_apo_df.PDB_ID))
    cls_df = pd.read_csv("../Output/Final/GPCR_PDB_classification.csv")[['UniProt_ID', 'Ant_PDB']]
    lig_info = pd.read_csv("../Input/Binding_site/PDB_ago_ant_chain_info_v2.csv", dtype={'PDB_ID': str})
    rep_chain = pd.read_csv("../Output/Binding_Residue/Rep_GPCR_chain.csv", dtype={'PDB_ID': str})

    dr_dict, summary = {}, []
    for _, row in cls_df[cls_df.Ant_PDB > 0].iterrows():
        uid = row['UniProt_ID']
        apo_source = rep_map.get(uid)
        apo_info = None
        if apo_source:
            sub = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == apo_source)]
            if not sub.empty:
                apo_chain = sub.loc[sub['score'].astype(float).idxmax(), 'chain_id']
                apo_info = process_structure(uid, pdb_id=apo_source, chain_id=apo_chain, is_apo=True)
        if apo_info is None:
            af_file = os.path.join(AF_DIR, f"AF-{uid}-F1-model_v3.pdb")
            apo_info = process_structure(uid, af_path=af_file, chain_id='A', is_apo=True)
        if apo_info is None:
            continue

        coords_acc = {}
        for _, ag in lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == 'antagonist')].iterrows():
            sub2 = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == ag.PDB_ID)]
            if sub2.empty:
                continue
            ag_chain = sub2.loc[sub2['score'].astype(float).idxmax(), 'chain_id']
            ag_info = process_structure(uid, pdb_id=ag.PDB_ID, chain_id=ag_chain, ligand_id=ag.LIGAND_ID, is_apo=False)
            if ag_info is None:
                continue
            common = np.intersect1d(apo_info['uni_idxs'], ag_info['uni_idxs'])
            if len(common) == 0:
                continue
            map_apo = dict(zip(apo_info['uni_idxs'], apo_info['coords']))
            map_ag = dict(zip(ag_info['uni_idxs'], ag_info['coords']))
            rmsd, disp, transformed = compute_rmsd_and_disp(
                np.array([map_apo[u] for u in common]),
                np.array([map_ag[u] for u in common])
            )
            diffs = [int(u) for u, d in zip(common, disp) if d >= DISP_THRESHOLD_ANT]
            summary.append({
                'UniProt_ID': uid,
                'Apo_Source': apo_source or 'AF2',
                'Antagonist_PDB': ag.PDB_ID,
                'Common_Residues': list(common),
                'Differential_Binding_Residues': diffs,
                'Overall_RMSD': rmsd
            })
            for u, coord in zip(common, transformed):
                if np.linalg.norm(map_apo[u] - coord) >= DISP_THRESHOLD_ANT:
                    coords_acc.setdefault(u, []).append(coord)
        if coords_acc:
            final_res = sorted(coords_acc.keys())
            avg_coords = [np.mean(np.vstack(coords_acc[u]), axis=0) for u in final_res]
            dr_dict[uid] = {
                'apo_pdb': apo_source or 'AF2',
                'antagonist_drs': final_res,
                'antagonist_dr_coords': torch.tensor(np.vstack(avg_coords), dtype=torch.float32)
            }

    pd.DataFrame(summary).to_csv(CSV_OUTPUT_PATH_ANT, index=False)
    torch.save(dr_dict, DR_OUTPUT_PATH_ANT)
    print(f"[INFO] DR-Ant saved to {CSV_OUTPUT_PATH_ANT} and {DR_OUTPUT_PATH_ANT}")
    return dr_dict

def extract_dr_state():
    """Extract DR-State by comparing Agonist-bound vs. Antagonist-bound structures."""
    cls_df = pd.read_csv("../Output/Final/GPCR_PDB_classification.csv")[['UniProt_ID', 'Ago_PDB', 'Ant_PDB']]
    lig_info = pd.read_csv("../Input/Binding_site/PDB_ago_ant_chain_info_v2.csv", dtype={'PDB_ID': str})
    rep_chain = pd.read_csv("../Output/Binding_Residue/Rep_GPCR_chain.csv", dtype={'PDB_ID': str})

    dr_dict, summary = {}, []
    for _, row in cls_df[(cls_df.Ago_PDB > 0) & (cls_df.Ant_PDB > 0)].iterrows():
        uid = row['UniProt_ID']
        ago_pdbs = lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == 'agonist')]['PDB_ID'].unique()
        ant_pdbs = lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == 'antagonist')]['PDB_ID'].unique()
        coords_acc = {}
        for pdb_ago in ago_pdbs:
            sub_ago = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == pdb_ago)]
            if sub_ago.empty:
                continue
            chain_ago = sub_ago.loc[sub_ago['score'].astype(float).idxmax(), 'chain_id']
            lig_row = lig_info[(lig_info.Entry == uid) & (lig_info.PDB_ID == pdb_ago) & (lig_info.MoA.str.lower() == 'agonist')].iloc[0]
            ago_info = process_structure(uid, pdb_id=pdb_ago, chain_id=chain_ago, ligand_id=lig_row.LIGAND_ID, is_apo=False)
            if ago_info is None:
                continue
            for pdb_ant in ant_pdbs:
                sub_ant = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == pdb_ant)]
                if sub_ant.empty:
                    continue
                chain_ant = sub_ant.loc[sub_ant['score'].astype(float).idxmax(), 'chain_id']
                lig_row2 = lig_info[(lig_info.Entry == uid) & (lig_info.PDB_ID == pdb_ant) & (lig_info.MoA.str.lower() == 'antagonist')].iloc[0]
                ant_info = process_structure(uid, pdb_id=pdb_ant, chain_id=chain_ant, ligand_id=lig_row2.LIGAND_ID, is_apo=False)
                if ant_info is None:
                    continue
                common = np.intersect1d(ago_info['uni_idxs'], ant_info['uni_idxs'])
                if len(common) < 3:
                    continue
                map_ago = dict(zip(ago_info['uni_idxs'], ago_info['coords']))
                map_ant = dict(zip(ant_info['uni_idxs'], ant_info['coords']))
                rmsd, disp, transformed = compute_rmsd_and_disp(
                    np.array([map_ago[u] for u in common]),
                    np.array([map_ant[u] for u in common])
                )
                diffs = [int(u) for u, d in zip(common, disp) if d >= DISP_THRESHOLD_STATE]
                summary.append({
                    'UniProt_ID': uid,
                    'Agonist_PDB': pdb_ago,
                    'Antagonist_PDB': pdb_ant,
                    'Common_Residues': list(common),
                    'Differential_Binding_Residues': diffs,
                    'Overall_RMSD': rmsd
                })
                for u, coord in zip(common, transformed):
                    if np.linalg.norm(map_ago[u] - coord) >= DISP_THRESHOLD_STATE:
                        coords_acc.setdefault(u, []).append(coord)
        if coords_acc:
            final_res = sorted(coords_acc.keys())
            avg_coords = [np.mean(np.vstack(coords_acc[u]), axis=0) for u in final_res]
            dr_dict[uid] = {
                'state_drs': final_res,
                'state_dr_coords': torch.tensor(np.vstack(avg_coords), dtype=torch.float32)
            }

    pd.DataFrame(summary).to_csv(CSV_OUTPUT_PATH_STATE, index=False)
    torch.save(dr_dict, DR_OUTPUT_PATH_STATE)
    print(f"[INFO] DR-State saved to {CSV_OUTPUT_PATH_STATE} and {DR_OUTPUT_PATH_STATE}")
    return dr_dict

def aggregate_dr_sets():
    """Aggregate DR-Ago, DR-Ant, DR-State into a single CSV with union of residues per GPCR."""
    def load_and_aggregate(csv_path, id_col="UniProt_ID", diff_col="Differential_Binding_Residues"):
        df = pd.read_csv(csv_path)
        df[diff_col] = df[diff_col].apply(ast.literal_eval)
        agg = df.groupby(id_col)[diff_col].agg(lambda lists: sorted({res for sub in lists for res in sub}))
        return agg

    dr_ago = load_and_aggregate(CSV_OUTPUT_PATH_AGO)
    dr_ant = load_and_aggregate(CSV_OUTPUT_PATH_ANT)
    dr_state = load_and_aggregate(CSV_OUTPUT_PATH_STATE)
    all_ids = set(dr_ago.index) | set(dr_ant.index) | set(dr_state.index)

    rows = []
    for uid in sorted(all_ids):
        rows.append({
            "UniProt_ID": uid,
            "DR-Ago": dr_ago.get(uid, []),
            "DR-Ant": dr_ant.get(uid, []),
            "DR-State": dr_state.get(uid, [])
        })

    df_dr = pd.DataFrame(rows)
    df_dr.to_csv(DR_SETS_CSV, index=False)
    print(f"[INFO] Aggregated DR sets saved to {DR_SETS_CSV}")
    return df_dr

def merge_dr_coordinates():
    """Merge DR-Ago, DR-Ant, DR-State coordinates into a single PyTorch dictionary."""
    ago_dict = torch.load(DR_OUTPUT_PATH_AGO)
    ant_dict = torch.load(DR_OUTPUT_PATH_ANT)
    state_dict = torch.load(DR_OUTPUT_PATH_STATE)
    all_keys = set(ago_dict.keys()) | set(ant_dict.keys()) | set(state_dict.keys)

    merged = {}
    for uid in sorted(all_keys):
        apo_pdb = None
        for d in (ago_dict, ant_dict, state_dict):
            if uid in d and "apo_pdb" in d[uid]:
                apo_pdb = d[uid]["apo_pdb"]
                break
        agonist_drs = ago_dict.get(uid, {}).get("agonist_drs", [])
        agonist_dr_coords = ago_dict.get(uid, {}).get("agonist_dr_coords", None)
        antagonist_drs = ant_dict.get(uid, {}).get("antagonist_drs", [])
        antagonist_dr_coords = ant_dict.get(uid, {}).get("antagonist_dr_coords", None)
        state_drs = state_dict.get(uid, {}).get("state_drs", [])
        state_dr_coords = state_dict.get(uid, {}).get("state_dr_coords", None)

        if not (agonist_drs or antagonist_drs or state_drs):
            continue

        merged[uid] = {
            "apo_pdb": apo_pdb,
            "agonist_drs": agonist_drs,
            "agonist_dr_coords": agonist_dr_coords,
            "antagonist_drs": antagonist_drs,
            "antagonist_dr_coords": antagonist_dr_coords,
            "state_drs": state_drs,
            "state_dr_coords": state_dr_coords
        }

    torch.save(merged, DR_MERGED_PATH)
    print(f"[INFO] Merged DR coordinates saved to {DR_MERGED_PATH}")
    return merged

def debug_dr_for(uid, dr_type="ago"):
    """Debug DR extraction for a specific UniProt ID and DR type (ago, ant, state)."""
    if dr_type not in ["ago", "ant", "state"]:
        raise ValueError("dr_type must be 'ago', 'ant', or 'state'")

    rep_apo_df = pd.read_csv("../Output/Final/Representative_Apo_Structures.csv", dtype={'PDB_ID': str})
    rep_apo_df = rep_apo_df[rep_apo_df.Binding_Coverage.astype(float) >= 50]
    rep_map = dict(zip(rep_apo_df.UniProt_ID, rep_apo_df.PDB_ID))
    rep_chain = pd.read_csv("../Output/Binding_Residue/Rep_GPCR_chain.csv", dtype={'PDB_ID': str})
    lig_info = pd.read_csv("../Input/Binding_site/PDB_ago_ant_chain_info_v2.csv", dtype={'PDB_ID': str})

    if dr_type in ["ago", "ant"]:
        apo_source = rep_map.get(uid)
        apo_info = None
        if apo_source:
            sub = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == apo_source)]
            if sub.empty:
                print(f"[{uid}] SKIP: No chain info for Apo {apo_source}")
                return
            apo_chain = sub.loc[sub['score'].astype(float).idxmax(), 'chain_id']
            apo_info = process_structure(uid, pdb_id=apo_source, chain_id=apo_chain, is_apo=True)
        if apo_info is None:
            af_file = os.path.join(AF_DIR, f"AF-{uid}-F1-model_v3.pdb")
            apo_info = process_structure(uid, af_path=af_file, chain_id='A', is_apo=True)
        if apo_info is None:
            print(f"[{uid}] SKIP: Failed to process Apo/AF2 structure")
            return

        moa = "agonist" if dr_type == "ago" else "antagonist"
        threshold = DISP_THRESHOLD_AGO if dr_type == "ago" else DISP_THRESHOLD_ANT
        rows = lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == moa)]
        if rows.empty:
            print(f"[{uid}] SKIP: No {moa} entries found")
            return

        for _, ag in rows.iterrows():
            pdb_id = ag.PDB_ID
            log = []
            sub2 = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == pdb_id)]
            if sub2.empty:
                log.append(f"no chain info for {moa}")
                print(f"[{uid}][{pdb_id}] {' | '.join(log)}")
                continue
            ag_chain = sub2.loc[sub2['score'].astype(float).idxmax(), 'chain_id']
            ag_info = process_structure(uid, pdb_id=pdb_id, chain_id=ag_chain, ligand_id=ag.LIGAND_ID, is_apo=False)
            if ag_info is None:
                log.append(f"process_structure failed on {moa}")
                print(f"[{uid}][{pdb_id}] {' | '.join(log)}")
                continue
            common = np.intersect1d(apo_info['uni_idxs'], ag_info['uni_idxs'])
            if len(common) == 0:
                log.append("no common binding residues")
                print(f"[{uid}][{pdb_id}] {' | '.join(log)}")
                continue
            map_apo = dict(zip(apo_info['uni_idxs'], apo_info['coords']))
            map_ag = dict(zip(ag_info['uni_idxs'], ag_info['coords']))
            rmsd, disp, _ = compute_rmsd_and_disp(
                np.array([map_apo[u] for u in common]),
                np.array([map_ag[u] for u in common])
            )
            diffs = [u for u, d in zip(common, disp) if d >= threshold]
            if not diffs:
                log.append(f"all disps < {threshold}Å")
                print(f"[{uid}][{pdb_id}] {' | '.join(log)}")
                continue
            print(f"[{uid}][{pdb_id}] PASS: Found DR residues {diffs}")

    elif dr_type == "state":
        ago_pdbs = lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == 'agonist')]['PDB_ID'].unique()
        ant_pdbs = lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == 'antagonist')]['PDB_ID'].unique()
        if len(ago_pdbs) == 0 or len(ant_pdbs) == 0:
            print(f"[{uid}] SKIP: Missing agonist or antagonist PDBs")
            return
        for pdb_ago in ago_pdbs:
            sub_ago = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == pdb_ago)]
            if sub_ago.empty:
                print(f"[{uid}][{pdb_ago}] SKIP: No chain info for agonist")
                continue
            chain_ago = sub_ago.loc[sub_ago['score'].astype(float).idxmax(), 'chain_id']
            lig_row = lig_info[(lig_info.Entry == uid) & (lig_info.PDB_ID == pdb_ago) & (lig_info.MoA.str.lower() == 'agonist')].iloc[0]
            ago_info = process_structure(uid, pdb_id=pdb_ago, chain_id=chain_ago, ligand_id=lig_row.LIGAND_ID, is_apo=False)
            if ago_info is None:
                print(f"[{uid}][{pdb_ago}] SKIP: Failed to process agonist")
                continue
            for pdb_ant in ant_pdbs:
                sub_ant = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == pdb_ant)]
                if sub_ant.empty:
                    print(f"[{uid}][{pdb_ago} vs {pdb_ant}] SKIP: No chain info for antagonist")
                    continue
                chain_ant = sub_ant.loc[sub_ant['score'].astype(float).idxmax(), 'chain_id']
                lig_row2 = lig_info[(lig_info.Entry == uid) & (lig_info.PDB_ID == pdb_ant) & (lig_info.MoA.str.lower() == 'antagonist')].iloc[0]
                ant_info = process_structure(uid, pdb_id=pdb_ant, chain_id=chain_ant, ligand_id=lig_row2.LIGAND_ID, is_apo=False)
                if ant_info is None:
                    print(f"[{uid}][{pdb_ago} vs {pdb_ant}] SKIP: Failed to process antagonist")
                    continue
                common = np.intersect1d(ago_info['uni_idxs'], ant_info['uni_idxs'])
                if len(common) < 3:
                    print(f"[{uid}][{pdb_ago} vs {pdb_ant}] SKIP: Too few common residues ({len(common)})")
                    continue
                map_ago = dict(zip(ago_info['uni_idxs'], ago_info['coords']))
                map_ant = dict(zip(ant_info['uni_idxs'], ant_info['coords']))
                rmsd, disp, _ = compute_rmsd_and_disp(
                    np.array([map_ago[u] for u in common]),
                    np.array([map_ant[u] for u in common])
                )
                diffs = [u for u, d in zip(common, disp) if d >= DISP_THRESHOLD_STATE]
                if not diffs:
                    print(f"[{uid}][{pdb_ago} vs {pdb_ant}] SKIP: All disps < {DISP_THRESHOLD_STATE}Å")
                    continue
                print(f"[{uid}][{pdb_ago} vs {pdb_ant}] PASS: Found DR residues {diffs}")

if __name__ == "__main__":
    print("Extracting DR-Ago...")
    extract_dr_ago()
    print("Extracting DR-Ant...")
    extract_dr_ant()
    print("Extracting DR-State...")
    extract_dr_state()
    print("Aggregating DR sets...")
    aggregate_dr_sets()
    print("Merging DR coordinates...")
    merge_dr_coordinates()
    print("DR extraction complete.")