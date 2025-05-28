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