import os
import re
import pandas as pd
from Bio.PDB import MMCIFParser, Polypeptide, is_aa
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from parse_gpcr_structures import *

def select_apo_structures(gpcr_pdb_csv, gpcr_bs_csv, rep_chain_csv):
    """Generate Representative_Apo_Structures_v2.csv."""
    gpcr_pdb_df = pd.read_csv(gpcr_pdb_csv)
    gpcr_bs_df = pd.read_csv(gpcr_bs_csv)
    rep_chain_df = pd.read_csv(rep_chain_csv)
    gpcr_pdb_df = gpcr_pdb_df[gpcr_pdb_df["Classification"].str.lower() == "apo"]
    results = []
    for uniprot_id, group in gpcr_pdb_df.groupby("Entry"):
        binding_row = gpcr_bs_df[gpcr_bs_df["UniProt_ID"] == uniprot_id]
        if binding_row.empty:
            continue
        binding_residues_list = binding_row.iloc[0]["Binding_Residues"]
        binding_residues_list = ast.literal_eval(binding_residues_list) if isinstance(binding_residues_list, str) else binding_residues_list
        candidates = []
        for _, row in group.iterrows():
            pdb_id = row["PDB_ID"].strip()
            cif_path = download_mmcif(pdb_id)
            if not cif_path:
                continue
            structure = parser.get_structure(pdb_id.lower(), cif_path)
            if not structure:
                continue
            optimal_chain_id = rep_chain_df[(rep_chain_df["UniProt_ID"] == uniprot_id) &
                                           (rep_chain_df["PDB_ID"] == pdb_id)]["chain_id"].iloc[0]
            model = next(structure.get_models())
            if optimal_chain_id not in model.child_dict:
                continue
            chain = model[optimal_chain_id]
            polypep = Polypeptide.Polypeptide(chain)
            pdb_seq = str(polypep.get_sequence())
            pdb_res_list = [res for res in chain if is_aa(res, standard=True)]
            uni_seq = fetch_uniprot_sequence(uniprot_id)
            if not uni_seq:
                continue
            aln = local_align_smith_waterman(pdb_seq, uni_seq)
            res_map = map_pdb_res_to_uniprot(pdb_res_list, aln["aligned_pdb_seq"], aln["aligned_uni_seq"])
            mapped_uniprot_set = set(res_map.values())
            overlap = set(binding_residues_list).intersection(mapped_uniprot_set)
            binding_coverage = len(overlap) / len(binding_residues_list) if binding_residues_list else 0
            binding_include = 1 if binding_coverage == 1 else 0
            resolution = extract_resolution_from_cif(cif_path)
            res_value = float(resolution) if resolution != "NA" else None
            candidates.append({
                "UniProt_ID": uniprot_id, "PDB_ID": pdb_id, "Binding_include": binding_include,
                "Binding_coverage": binding_coverage, "Resolution": resolution, "Res_value": res_value
            })
        if candidates:
            bind_incl_candidates = [c for c in candidates if c["Binding_include"] == 1]
            if bind_incl_candidates:
                with_resolution = [c for c in bind_incl_candidates if c["Res_value"] is not None]
                best = min(with_resolution, key=lambda x: x["Res_value"]) if with_resolution else bind_incl_candidates[0]
            else:
                coverage_sorted = sorted(candidates, key=lambda x: x["Binding_coverage"], reverse=True)
                max_cov = coverage_sorted[0]["Binding_coverage"]
                top_cov = [c for c in coverage_sorted if c["Binding_coverage"] == max_cov]
                with_resolution = [c for c in top_cov if c["Res_value"] is not None]
                best = min(with_resolution, key=lambda x: x["Res_value"]) if with_resolution else top_cov[0]
            results.append({"UniProt_ID": uniprot_id, "PDB_ID": best["PDB_ID"], "Resolution": best["Resolution"]})
    df = pd.DataFrame(results, columns=["UniProt_ID", "PDB_ID", "Resolution"])
    df.to_csv(os.path.join(OUTPUT_DIR, "Representative_Apo_Structures_v2.csv"), index=False)
    return df

def find_best_chain_local(structure, seq_uniprot,
                          cov_chain_thr=0.7, cov_uniprot_thr=0.7, identity_thr=0.7):
    """
    Among all chains in the first model of 'structure', perform local alignment
    with 'seq_uniprot' and compute coverageChain, coverageUniProt, identity, alignment score.
    Return (best_chain_id, df_chain_info).
    
    - best_chain_id: chain ID that passes thresholds and has highest alignment score.
      If none pass, return None.
    - df_chain_info: DataFrame with columns:
        chain_id, chain_length, score, coverageChain, coverageUniProt, identity, pass_filter
    """
    model = next(structure.get_models())
    records = []

    for chain in model:
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

    import pandas as pd
    df_chain = pd.DataFrame(records)
    if df_chain.empty:
        return None, pd.DataFrame()

    # Define pass_filter
    df_chain["pass_filter"] = (
        (df_chain["coverageChain"] >= cov_chain_thr) &
        (df_chain["coverageUniProt"] >= cov_uniprot_thr) &
        (df_chain["identity"] >= identity_thr)
    )

    # Among those passing, pick the chain with highest score
    df_pass = df_chain[df_chain["pass_filter"]]
    if df_pass.empty:
        best_chain_id = None
    else:
        df_pass = df_pass.sort_values("score", ascending=False)
        best_chain_id = df_pass.iloc[0]["chain_id"]

    return best_chain_id, df_chain

def process_gpcr_pdb_chains(df_uniprot_acs, cov_chain_thr=0.7, cov_uniprot_thr=0.7, identity_thr=0.7):
    """
    For each row in df_uniprot_acs, we:
      1) Fetch the UniProt sequence
      2) For each PDB ID (in the 'PDB' column, separated by ';'), download the .cif file
      3) Parse the .cif with BioPython's MMCIFParser
      4) Find the best chain that meets coverage_thr & identity_thr
      5) Collect results in a DataFrame
    
    Returns a DataFrame with columns:
      [ UniProt_ID, PDB_ID, chain_id, chain_length, score, coverage, identity, pass_filter ]
    """
    results = []

    for idx, row in df_uniprot_acs.iterrows():
        uniprot_id = str(row["Entry"]).strip()
        pdb_list_str = row.get("PDB", "")
        if pd.isna(pdb_list_str) or not pdb_list_str:
            continue

        # 1) Fetch UniProt sequence
        uni_seq = fetch_uniprot_sequence(uniprot_id)
        if not uni_seq:
            print(f"[WARNING] UniProt seq not found for {uniprot_id}. Skipping.")
            continue

        # PDB column can be something like "8GO9;8J8V;8J8Z;"
        pdb_ids = [p.strip() for p in pdb_list_str.split(';') if p.strip()]

        for pdb_id in pdb_ids:
            pdb_id_lower = pdb_id.lower()

            # 2) Download .cif
            cif_file_path = download_mmcif(pdb_id_lower, CIF_DOWNLOAD_DIR)
            if not cif_file_path:
                continue

            # 3) Parse structure
            try:
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure(pdb_id_lower, cif_file_path)
            except Exception as e:
                print(f"[ERROR] Failed to parse CIF {pdb_id}: {e}")
                continue

            # 4) Find best chain
            best_chain_id, df_chain = find_best_chain_local(
                structure, uni_seq
            )

            # 5) Collect results
            if not df_chain.empty:
                df_chain["UniProt_ID"] = uniprot_id
                df_chain["PDB_ID"] = pdb_id
                results.append(df_chain)

    if not results:
        return pd.DataFrame()
    df_res = pd.concat(results, ignore_index=True)
    # Reorder columns for clarity
    cols = ["UniProt_ID", "PDB_ID", "chain_id", "chain_length", "score", "coverageChain", "coverageUniProt", "identity", "pass_filter"]
    df_res = df_res[cols]
    return df_res
