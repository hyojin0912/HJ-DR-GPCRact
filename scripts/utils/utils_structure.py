"""
utils_structure.py

A utility script for processing and analyzing GPCR (G protein-coupled receptor)
structures from PDB files. This script is designed to be run as a command-line
tool to perform a two-step pipeline:

1.  **Identify Representative Chains**: For a given list of GPCRs and their
    associated PDB entries, this script finds the most suitable protein chain
    by aligning it with the UniProt sequence. The results are saved to a CSV file.

2.  **Select Representative Apo Structures**: Using the chain information from
    step 1, this script identifies the best "Apo" (ligand-free) structure for
    each GPCR based on binding site coverage and experimental resolution.

This script is a prerequisite for downstream structural modeling and analysis.
"""

import os
import ast
import argparse
import re
import pandas as pd
from Bio.PDB import MMCIFParser, Polypeptide, is_aa
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from parse_gpcr_structures import *

# --- Helper Functions ---
def _find_best_candidate(candidates: list) -> dict:
    """
    Selects the best PDB candidate from a list based on a two-tier criterion.
    1. Prioritize candidates with full binding site coverage. Among these,
       select the one with the best (lowest) resolution.
    2. If none have full coverage, select candidates with the highest partial
       coverage. Among these, select the one with the best resolution.

    Args:
        candidates (list): A list of dictionaries, where each dictionary
                           represents a PDB candidate and its properties.

    Returns:
        dict: The dictionary of the best candidate found.
    """
    # Tier 1: Prefer candidates with full binding site coverage
    full_coverage_candidates = [c for c in candidates if c["Binding_include"] == 1]
    if full_coverage_candidates:
        with_resolution = [c for c in full_coverage_candidates if c["Res_value"] is not None]
        if with_resolution:
            return min(with_resolution, key=lambda x: x["Res_value"])
        return full_coverage_candidates[0]

    # Tier 2: Fallback to highest partial coverage
    sorted_by_coverage = sorted(candidates, key=lambda x: x["Binding_coverage"], reverse=True)
    max_coverage = sorted_by_coverage[0]["Binding_coverage"]
    top_coverage_candidates = [c for c in sorted_by_coverage if c["Binding_coverage"] == max_coverage]

    with_resolution = [c for c in top_coverage_candidates if c["Res_value"] is not None]
    if with_resolution:
        return min(with_resolution, key=lambda x: x["Res_value"])
    return top_coverage_candidates[0]


# --- Core Processing Functions ---

def find_best_chain_local(structure, seq_uniprot: str, cov_chain_thr=0.7, cov_uniprot_thr=0.7, identity_thr=0.7):
    """
    Finds the best matching chain in a PDB structure against a UniProt sequence.

    It performs local alignment for all chains and returns the ID of the chain
    that best passes coverage and identity thresholds with the highest score.

    Args:
        structure (Bio.PDB.Structure.Structure): The PDB structure object.
        seq_uniprot (str): The UniProt amino acid sequence.
        cov_chain_thr (float): Minimum coverage threshold for the PDB chain.
        cov_uniprot_thr (float): Minimum coverage threshold for the UniProt sequence.
        identity_thr (float): Minimum identity threshold.

    Returns:
        tuple[str | None, pd.DataFrame]: A tuple containing:
            - The ID of the best chain, or None if none pass the filters.
            - A DataFrame with alignment details for all chains.
    """
    model = next(structure.get_models())
    records = []

    for chain in model:
        try:
            polypep = Polypeptide.Polypeptide(chain)
            seq_chain = str(polypep.get_sequence())
            if not seq_chain:
                continue

            res = local_align_smith_waterman(seq_chain, seq_uniprot)
            rec = {
                "chain_id": chain.id,
                "chain_length": len(seq_chain),
                "score": res["score"],
                "coverageChain": res["coverageChain"],
                "coverageUniProt": res["coverageUniProt"],
                "identity": res["identity"]
            }
            records.append(rec)
        except Exception:
            continue
            
    if not records:
        return None, pd.DataFrame()

    df_chain = pd.DataFrame(records)
    df_chain["pass_filter"] = (
        (df_chain["coverageChain"] >= cov_chain_thr) &
        (df_chain["coverageUniProt"] >= cov_uniprot_thr) &
        (df_chain["identity"] >= identity_thr)
    )

    df_pass = df_chain[df_chain["pass_filter"]].sort_values("score", ascending=False)

    best_chain_id = df_pass.iloc[0]["chain_id"] if not df_pass.empty else None
    return best_chain_id, df_chain


def process_gpcr_pdb_chains(gpcr_info_csv: str, output_dir: str, cif_cache_dir: str) -> str:
    """
    Identifies the most representative chain for each GPCR-PDB entry.

    Args:
        gpcr_info_csv (str): Path to the input CSV file containing 'Entry' (UniProt ID)
                             and 'PDB' columns.
        output_dir (str): Directory to save the output CSV file.
        cif_cache_dir (str): Directory to download and cache mmCIF files.

    Returns:
        str: The path to the generated output file.
    """
    df_uniprot_acs = pd.read_csv(gpcr_info_csv)
    results = []
    parser = MMCIFParser(QUIET=True)

    print("[INFO] Finding representative chains for each GPCR-PDB pair...")
    for _, row in tqdm(df_uniprot_acs.iterrows(), total=len(df_uniprot_acs)):
        uniprot_id = str(row["Entry"]).strip()
        pdb_list_str = row.get("PDB", "")
        if pd.isna(pdb_list_str) or not pdb_list_str:
            continue

        uni_seq = fetch_uniprot_sequence(uniprot_id)
        if not uni_seq:
            print(f"[WARN] UniProt sequence not found for {uniprot_id}. Skipping.")
            continue

        pdb_ids = [p.strip() for p in pdb_list_str.split(';') if p.strip()]
        for pdb_id in pdb_ids:
            cif_file_path = download_mmcif(pdb_id, cif_cache_dir)
            if not cif_file_path:
                continue

            try:
                structure = parser.get_structure(pdb_id.lower(), cif_file_path)
                _, df_chain = find_best_chain_local(structure, uni_seq)
                if not df_chain.empty:
                    df_chain["UniProt_ID"] = uniprot_id
                    df_chain["PDB_ID"] = pdb_id
                    results.append(df_chain)
            except Exception as e:
                print(f"[ERROR] Failed to process {pdb_id} for {uniprot_id}: {e}")
                continue

    if not results:
        print("[WARN] No valid chains were processed.")
        return ""
    
    df_res = pd.concat(results, ignore_index=True)
    cols = ["UniProt_ID", "PDB_ID", "chain_id", "chain_length", "score", "coverageChain", "coverageUniProt", "identity", "pass_filter"]
    df_res = df_res[cols]
    
    output_path = os.path.join(output_dir, "Rep_GPCR_chain.csv")
    df_res.to_csv(output_path, index=False)
    print(f"[SUCCESS] Representative chain data saved to {output_path}")
    return output_path


def select_apo_structures(
    pdb_classification_csv: str,
    binding_residues_csv: str,
    rep_chain_csv: str,
    output_dir: str,
    cif_cache_dir: str
) -> str:
    """
    Selects the best representative "Apo" structure for each GPCR.
    This version uses a master classification file directly.

    Args:
        pdb_classification_csv (str): Path to the master CSV file containing PDB_ID,
                                      Entry (UniProt ID), and Classification.
        binding_residues_csv (str): Path to CSV with binding site residues for each UniProt ID.
        rep_chain_csv (str): Path to CSV mapping each PDB to its representative chain.
        output_dir (str): Directory to save the output CSV file.
        cif_cache_dir (str): Directory to cache downloaded mmCIF files.
    """
    # --- Load Data ---
    pdb_classification_df = pd.read_csv(pdb_classification_csv)
    binding_residues_df = pd.read_csv(binding_residues_csv)
    rep_chain_df = pd.read_csv(rep_chain_csv)
    parser = MMCIFParser(QUIET=True)

    # --- Filter for Apo structures directly from the classification file ---
    apo_df = pdb_classification_df[pdb_classification_df["Classification"].str.lower() == "apo"].copy()

    # --- Main Logic ---
    results = []
    print("[INFO] Selecting best Apo structure for each GPCR...")
    for uniprot_id, group in tqdm(apo_df.groupby("Entry"), total=apo_df['Entry'].nunique()):
        binding_row = binding_residues_df[binding_residues_df["UniProt_ID"] == uniprot_id]
        if binding_row.empty:
            continue
        # Use ast.literal_eval to safely evaluate the string representation of the list
        binding_residues_list = ast.literal_eval(binding_row.iloc[0]["Binding_Residues"])

        candidates = []
        for _, row in group.iterrows():
            pdb_id = row["PDB_ID"].strip()
            chain_row = rep_chain_df[(rep_chain_df["UniProt_ID"] == uniprot_id) & (rep_chain_df["PDB_ID"] == pdb_id)]
            if chain_row.empty:
                continue
            optimal_chain_id = chain_row.iloc[0]["chain_id"]
            
            cif_path = download_mmcif(pdb_id, cif_cache_dir)
            if not cif_path:
                continue
            
            try:
                structure = parser.get_structure(pdb_id.lower(), cif_path)
                model = next(structure.get_models())
                if optimal_chain_id not in model:
                    continue
                
                chain = model[optimal_chain_id]
                pdb_res_list = [res for res in chain if is_aa(res, standard=True)]
                pdb_seq = Polypeptide.Polypeptide(chain).get_sequence()
                
                uni_seq = fetch_uniprot_sequence(uniprot_id)
                if not uni_seq: continue

                aln = local_align_smith_waterman(str(pdb_seq), uni_seq)
                res_map = map_pdb_res_to_uniprot(pdb_res_list, aln["aligned_pdb_seq"], aln["aligned_uni_seq"])
                
                mapped_uniprot_set = set(res_map.values())
                overlap = set(binding_residues_list).intersection(mapped_uniprot_set)
                coverage = len(overlap) / len(binding_residues_list) if binding_residues_list else 0
                
                resolution = extract_resolution_from_cif(cif_path)
                res_value = float(resolution) if resolution != "NA" else None

                candidates.append({
                    "UniProt_ID": uniprot_id, "PDB_ID": pdb_id, "Binding_include": int(coverage == 1.0),
                    "Binding_coverage": coverage, "Resolution": resolution, "Res_value": res_value
                })
            except Exception as e:
                print(f"[ERROR] Failed to process Apo structure {pdb_id} for {uniprot_id}: {e}")
                continue
                
        if candidates:
            best_candidate = _find_best_candidate(candidates)
            if best_candidate:
                results.append({
                    "UniProt_ID": best_candidate["UniProt_ID"],
                    "PDB_ID": best_candidate["PDB_ID"],
                    "Resolution": best_candidate["Resolution"]
                })

    df = pd.DataFrame(results, columns=["UniProt_ID", "PDB_ID", "Resolution"])
    output_path = os.path.join(output_dir, "Representative_Apo_Structures.csv")
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Representative Apo structures saved to {output_path}")
    return output_path

# --- Main Execution Block ---

def main():
    """Main function to parse command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="GPCR Structure Analysis Pipeline: Identifies representative chains and Apo structures.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # This argument is only needed for the first step of the pipeline
    parser.add_argument(
        "--gpcr_info_csv", type=str, default="data/Human_GPCR_PDB_Info.csv",
        help="Path to the master CSV with UniProt IDs and all associated PDB IDs.\n(Used for Step 1: Chain Processing).\n(Default: data/Human_GPCR_PDB_Info.csv)"
    )
    # This is the corrected classification file for the second step
    parser.add_argument(
        "--pdb_classification_csv", type=str, default="data/GPCR_PDB_classify_v2.csv",
        help="Path to the CSV file classifying each PDB ID.\n(Requires 'Entry', 'PDB_ID', 'Classification' columns).\n(Used for Step 2: Apo Selection).\n(Default: data/GPCR_PDB_classify_v2.csv)"
    )
    parser.add_argument(
        "--binding_residues_csv", type=str, default="data/GPCR_Binding_Residues.UniProt.csv",
        help="Path to the CSV file with binding site residues for each UniProt ID.\n(Default: data/GPCR_Binding_Residues.UniProt.csv)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/structural_analysis",
        help="Directory to save the output files.\n(Default: outputs/structural_analysis)"
    )
    parser.add_argument(
        "--cif_cache_dir", type=str, default="cif_cache",
        help="Directory to cache downloaded mmCIF files.\n(Default: cif_cache)"
    )
    parser.add_argument(
        "--skip_chain_processing", action="store_true",
        help="Skip the 'process_gpcr_pdb_chains' step if its output already exists."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cif_cache_dir, exist_ok=True)

    # --- Pipeline Execution ---
    rep_chain_path = os.path.join(args.output_dir, "Rep_GPCR_chain.csv")

    if args.skip_chain_processing and os.path.exists(rep_chain_path):
        print(f"[INFO] Skipping chain processing. Using existing file: {rep_chain_path}")
    else:
        print("[INFO] === STEP 1: Processing GPCR PDB chains... ===")
        rep_chain_path = process_gpcr_pdb_chains(
            gpcr_info_csv=args.gpcr_info_csv,
            output_dir=args.output_dir,
            cif_cache_dir=args.cif_cache_dir
        )

    if not rep_chain_path or not os.path.exists(rep_chain_path):
        print("[ERROR] Representative chain file was not found or generated. Cannot proceed. Exiting.")
        return

    print("\n[INFO] === STEP 2: Selecting representative Apo structures... ===")
    select_apo_structures(
        pdb_classification_csv=args.pdb_classification_csv, # The main input for this step
        binding_residues_csv=args.binding_residues_csv,
        rep_chain_csv=rep_chain_path,
        output_dir=args.output_dir,
        cif_cache_dir=args.cif_cache_dir
    )

    print("\n[INFO] ✅ Pipeline finished successfully.")


if __name__ == "__main__":
    main()
