#!/usr/bin/env python
# coding: utf-8

"""
parse_gpcr_structures.py: Main pipeline for processing GPCR structures.
Downloads PDB/mmCIF/AlphaFold2 structures, extracts ligands, identifies binding sites,
maps residues to UniProt, and generates output files for DR analysis.
"""

import os
import ast
import pandas as pd
import requests
from Bio.PDB import PDBParser, MMCIFParser, Polypeptide, is_aa
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict

# Configuration
PDB_DOWNLOAD_DIR = "./Data/PDB_Files/"
CIF_DOWNLOAD_DIR = "./Data/CIF_Files/"
AF2_DOWNLOAD_DIR = "./Data/AF_PDB/"
OUTPUT_DIR = "./Output/Binding_Residue/"
os.makedirs(PDB_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CIF_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(AF2_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLOSUM62 = matlist.blosum62
GAP_OPEN_PENALTY = -10
GAP_EXTEND_PENALTY = -0.5

EXCLUDED_LIGAND_IDS = {
    "HOH", "WAT", "H2O", "W", "SO4", "NA", "CA", "K", "CL", "MG", "ZN", "MN", "CD",
    "GLU", "GOL", "PO4", "EDO", "PEG", "SEP", "TPO", "MSE", "GDP", "GTP", "GMP",
    "GNP", "GTPS", "ALF", "GDPS", "GDX", "AMP", "ADP", "ATP", "CAMP", "CGMP",
    "HEM", "FAD", "FMN", "NAD", "NDP", "NAP", "MPD"
}

# Initialize parsers
pdb_parser = PDBParser(QUIET=True)
cif_parser = MMCIFParser(QUIET=True)

# Utility functions
def download_pdb(pdb_id, download_dir=PDB_DOWNLOAD_DIR):
    """Download a PDB file from RCSB."""
    pdb_id = pdb_id.lower()
    pdb_file_path = os.path.join(download_dir, f"{pdb_id}.pdb")
    if os.path.exists(pdb_file_path):
        return pdb_file_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(pdb_file_path, 'wb') as f:
                f.write(response.content)
            print(f"[INFO] Downloaded {pdb_id}.pdb")
            return pdb_file_path
        print(f"[WARNING] Failed to download PDB {pdb_id}: HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"[ERROR] download_pdb({pdb_id}): {e}")
        return None

def download_mmcif(pdb_id, download_dir=CIF_DOWNLOAD_DIR):
    """Download an mmCIF file from RCSB."""
    pdb_id = pdb_id.lower()
    cif_file_path = os.path.join(download_dir, f"{pdb_id}.cif")
    if os.path.exists(cif_file_path):
        return cif_file_path
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(cif_file_path, 'wb') as f:
                f.write(response.content)
            print(f"[INFO] Downloaded {pdb_id}.cif")
            return cif_file_path
        print(f"[WARNING] Failed to download CIF {pdb_id}: HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"[ERROR] download_mmcif({pdb_id}): {e}")
        return None

def download_alphafold_structure(uniprot_ac, output_dir=AF2_DOWNLOAD_DIR):
    """Download an AlphaFold2-predicted PDB from EBI."""
    pdb_filename = f"AF-{uniprot_ac}-F1-model_v3.pdb"
    output_path = os.path.join(output_dir, pdb_filename)
    if os.path.exists(output_path):
        return output_path
    url = f"https://alphafold.ebi.ac.uk/files/{pdb_filename}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"[INFO] Downloaded AF2 structure {pdb_filename}")
            return output_path
        print(f"[WARNING] Failed to download AF2 {uniprot_ac}: HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"[ERROR] download_alphafold_structure({uniprot_ac}): {e}")
        return None

def parse_structure(pdb_id, file_path, parser_type="pdb"):
    """Parse a PDB or mmCIF file."""
    try:
        parser = pdb_parser if parser_type == "pdb" else cif_parser
        structure = parser.get_structure(pdb_id.lower(), file_path)
        return structure
    except Exception as e:
        print(f"[ERROR] parse_structure({pdb_id}): {e}")
        return None

def extract_resolution_from_cif(cif_path: str) -> float | None:
    """Extracts the resolution from an mmCIF file."""
    try:
        cif_dict = MMCIF2Dict(cif_path)
        keys_to_check = [
            "_refine.ls_d_res_high",
            "_reflns.d_resolution_high",
            "_em_3d_reconstruction.resolution"
        ]
        for key in keys_to_check:
            if key in cif_dict:
                value = cif_dict[key]
                # The value can be a list, take the first element.
                if isinstance(value, list):
                    value = value[0]
                # Attempt to convert to float
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue
        return None
    except Exception as e:
        print(f"[WARN] Could not parse or extract resolution from {os.path.basename(cif_path)}: {e}")
        return None

def fetch_uniprot_sequence(uniprot_id):
    """Fetch UniProt sequence as a string."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return "".join(l.strip() for l in r.text.split('\n') if not l.startswith('>') and l.strip())
        print(f"[WARNING] fetch_uniprot_sequence({uniprot_id}): HTTP {r.status_code}")
        return ""
    except Exception as e:
        print(f"[ERROR] fetch_uniprot_sequence({uniprot_id}): {e}")
        return ""

def local_align_smith_waterman(seq_chain, seq_uniprot):
    """Perform Smith-Waterman alignment."""
    alns = pairwise2.align.localds(seq_chain, seq_uniprot, BLOSUM62, GAP_OPEN_PENALTY, GAP_EXTEND_PENALTY)
    if not alns:
        return {"aligned_pdb_seq": "", "aligned_uni_seq": "", "score": 0.0, "coverageChain": 0.0,
                "coverageUniProt": 0.0, "identity": 0.0}
    aligned_pdb, aligned_uni, score, _, _ = alns[0]
    aligned_len_chain = sum(1 for c in aligned_pdb if c != '-')
    coverage_chain = aligned_len_chain / len(seq_chain) if seq_chain else 0.0
    coverage_uniprot = aligned_len_chain / len(seq_uniprot) if seq_uniprot else 0.0
    matches = sum(1 for c1, c2 in zip(aligned_pdb, aligned_uni) if c1 == c2 and c1 != '-')
    aligned_positions = sum(1 for c1, c2 in zip(aligned_pdb, aligned_uni) if c1 != '-' and c2 != '-')
    identity = matches / aligned_positions if aligned_positions > 0 else 0.0
    return {"aligned_pdb_seq": aligned_pdb, "aligned_uni_seq": aligned_uni, "score": score,
            "coverageChain": coverage_chain, "coverageUniProt": coverage_uniprot, "identity": identity}

def map_pdb_res_to_uniprot(pdb_res_list, aligned_pdb_seq, aligned_uni_seq):
    """Map PDB residues to UniProt indices."""
    res_map = {}
    pdb_idx = uni_idx = 0
    for cp, cu in zip(aligned_pdb_seq, aligned_uni_seq):
        if cp != '-':
            pdb_idx += 1
        if cu != '-':
            uni_idx += 1
        if cp != '-' and cu != '-' and pdb_idx <= len(pdb_res_list):
            res_map[pdb_res_list[pdb_idx - 1]] = uni_idx
    return res_map

def get_ligand_details(ligand_id):
    """Fetch SMILES and InChIKey for a ligand."""
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            smiles = data.get('rcsb_chem_comp_descriptor', {}).get('smiles')
            inchi_key = data.get('rcsb_chem_comp_descriptor', {}).get('InChIKey')
            if smiles and not inchi_key:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    inchi_key = AllChem.InchiToInchiKey(AllChem.MolToInchi(mol))
            return smiles, inchi_key
    except Exception as e:
        print(f"[WARNING] get_ligand_details({ligand_id}): {e}")
    return None, None

def extract_unified_ligands(structure, min_heavy_atoms=20):
    """Extract and unify HETATM residues into ligand groups."""
    model = next(structure.get_models())
    ligand_map = defaultdict(list)
    for chain in model:
        chain_id = chain.id
        for residue in chain:
            if residue.id[0] not in (' ', 'W') and residue.get_resname().upper() not in EXCLUDED_LIGAND_IDS:
                lig_name = residue.get_resname().upper()
                resseq = residue.id[1]
                key = (chain_id, lig_name, resseq)
                ligand_map[key].extend(atom for atom in residue.get_atoms())
    results = []
    for (chain_id, lig_name, resseq), atoms in ligand_map.items():
        heavy_count = sum(1 for atom in atoms if atom.element != 'H')
        if heavy_count < min_heavy_atoms:
            continue
        smiles, inchi_key = get_ligand_details(lig_name)
        results.append({
            "chain_id": chain_id, "ligand_id": lig_name, "resseq": resseq,
            "heavy_atom_count": heavy_count, "smiles": smiles, "inchi_key": inchi_key
        })
    return results

def get_ligand_by_id(structure, ligand_id):
    """Find ligand residue by ID."""
    ligand_id = ligand_id.upper()
    model = next(structure.get_models())
    for chain in model:
        for residue in chain:
            if residue.id[0] not in (' ', 'W') and residue.get_resname().upper() == ligand_id:
                return residue
    return None

def extract_binding_site(structure, ligand, chain_id, threshold=6.0):
    """Extract residues within threshold distance of ligand."""
    binding_res = set()
    if not ligand:
        return binding_res
    model = next(structure.get_models())
    if chain_id not in model.child_dict:
        return binding_res
    chain = model[chain_id]
    for atom_lig in ligand.get_atoms():
        for residue in chain:
            if is_aa(residue, standard=True):
                for atom_res in residue:
                    if atom_lig - atom_res < threshold:
                        binding_res.add(residue)
                        break
    return binding_res

# Main pipeline functions
def process_representative_chains(df_uniprot_acs, cov_chain_thr=0.7, cov_uniprot_thr=0.7, identity_thr=0.7):
    """Generate Rep_GPCR_chain.csv."""
    results = []
    for _, row in df_uniprot_acs.iterrows():
        uniprot_id = str(row["Entry"]).strip()
        pdb_list = [p.strip() for p in str(row.get("PDB", "")).split(';') if p.strip()]
        uni_seq = fetch_uniprot_sequence(uniprot_id)
        if not uni_seq or not pdb_list:
            continue
        for pdb_id in pdb_list:
            cif_path = download_mmcif(pdb_id)
            if not cif_path:
                continue
            structure = parse_structure(pdb_id, cif_path, "cif")
            if not structure:
                continue
            model = next(structure.get_models())
            for chain in model:
                polypep = Polypeptide.Polypeptide(chain)
                seq_chain = str(polypep.get_sequence())
                if not seq_chain:
                    continue
                aln = local_align_smith_waterman(seq_chain, uni_seq)
                pass_filter = (aln["coverageChain"] >= cov_chain_thr and
                               aln["coverageUniProt"] >= cov_uniprot_thr and
                               aln["identity"] >= identity_thr)
                results.append({
                    "UniProt_ID": uniprot_id, "PDB_ID": pdb_id, "chain_id": chain.id,
                    "chain_length": len(seq_chain), "score": aln["score"],
                    "coverageChain": aln["coverageChain"], "coverageUniProt": aln["coverageUniProt"],
                    "identity": aln["identity"], "pass_filter": pass_filter
                })
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "Rep_GPCR_chain.csv"), index=False)
    return df

def process_ligands(df_uniprot_acs):
    """Generate Rep_Ligand.csv."""
    results = []
    for _, row in df_uniprot_acs.iterrows():
        uniprot_ac = str(row["Entry"]).strip()
        pdb_list = [p.strip() for p in str(row.get("PDB", "")).split(';') if p.strip()]
        if not pdb_list:
            continue
        for pdb_id in pdb_list:
            cif_path = download_mmcif(pdb_id)
            if not cif_path:
                continue
            structure = parse_structure(pdb_id, cif_path, "cif")
            if not structure:
                continue
            ligands = extract_unified_ligands(structure)
            if not ligands:
                results.append([uniprot_ac, pdb_id, None, None, None, None, None, None])
            for lig in ligands:
                results.append([
                    uniprot_ac, pdb_id, lig["chain_id"], lig["ligand_id"], lig["resseq"],
                    lig["heavy_atom_count"], lig["smiles"], lig["inchi_key"]
                ])
    df = pd.DataFrame(results, columns=[
        "UNIPROT_AC", "PDB_ID", "chain_id", "ligand_id", "resseq",
        "heavy_atom_count", "smiles", "inchi_key"
    ])
    df.to_csv(os.path.join(OUTPUT_DIR, "Rep_Ligand.csv"), index=False)
    return df

def process_binding_residues(rep_chain_df, rep_ligand_df, dist_thresholds=(4.0, 6.0, 8.0)):
    """Generate PDB_Binding_Res.csv."""
    df_chain = rep_chain_df[rep_chain_df["pass_filter"] == True].copy()
    df_merged = pd.merge(df_chain, rep_ligand_df, how="inner",
                         left_on=["UniProt_ID", "PDB_ID"],
                         right_on=["UNIPROT_AC", "PDB_ID"])
    results = []
    for _, row in df_merged.iterrows():
        uniprot_id = row["UniProt_ID"]
        pdb_id = row["PDB_ID"]
        chain_id = row["chain_id"]
        ligand_id = row["ligand_id"]
        if pd.isna(ligand_id):
            continue
        cif_path = download_mmcif(pdb_id)
        if not cif_path:
            continue
        structure = parse_structure(pdb_id, cif_path, "cif")
        if not structure:
            continue
        lig_res = get_ligand_by_id(structure, ligand_id)
        if not lig_res:
            continue
        for th in dist_thresholds:
            bs_residues = extract_binding_site(structure, lig_res, chain_id, th)
            residue_list = [res.id[1] for res in bs_residues]
            results.append([uniprot_id, pdb_id, chain_id, ligand_id, th, len(bs_residues), residue_list])
    df = pd.DataFrame(results, columns=[
        "UniProt_ID", "PDB_ID", "chain_id", "ligand_id", "threshold",
        "n_binding_res", "binding_residues"
    ])
    df.to_csv(os.path.join(OUTPUT_DIR, "PDB_Binding_Res.csv"), index=False)
    return df

def map_binding_residues_to_uniprot(binding_res_df):
    """Generate PDB_Binding_Residues_UniProt.csv."""
    records = []
    for _, row in binding_res_df.iterrows():
        uniprot_id = row["UniProt_ID"]
        pdb_id = row["PDB_ID"]
        chain_id = row["chain_id"]
        ligand_id = row["ligand_id"]
        n_bind = row["n_binding_res"]
        binding_list = ast.literal_eval(row["binding_residues"]) if isinstance(row["binding_residues"], str) else row["binding_residues"]
        uni_seq = fetch_uniprot_sequence(uniprot_id)
        cif_path = download_mmcif(pdb_id)
        if not uni_seq or not cif_path:
            records.append([uniprot_id, pdb_id, chain_id, ligand_id, n_bind, binding_list, [], 0.0, 0.0, 0.0, 0.0])
            continue
        structure = parse_structure(pdb_id, cif_path, "cif")
        if not structure:
            continue
        model = next(structure.get_models())
        if chain_id not in model.child_dict:
            records.append([uniprot_id, pdb_id, chain_id, ligand_id, n_bind, binding_list, [], 0.0, 0.0, 0.0, 0.0])
            continue
        chain = model[chain_id]
        polypep = Polypeptide.Polypeptide(chain)
        pdb_seq = str(polypep.get_sequence())
        pdb_res_list = [res for res in chain if is_aa(res, standard=True)]
        aln = local_align_smith_waterman(pdb_seq, uni_seq)
        res_map = map_pdb_res_to_uniprot(pdb_res_list, aln["aligned_pdb_seq"], aln["aligned_uni_seq"])
        mapped_uniprot = [res_map[res] for res in pdb_res_list if res.id[1] in binding_list and res in res_map]
        records.append([
            uniprot_id, pdb_id, chain_id, ligand_id, n_bind, binding_list, mapped_uniprot,
            aln["coverageChain"], aln["coverageUniProt"], aln["identity"], aln["score"]
        ])
    df = pd.DataFrame(records, columns=[
        "UniProt_ID", "PDB_ID", "chain_id", "ligand_id", "n_binding_res",
        "binding_residues", "binding_res_uniprot", "coverageChain",
        "coverageUniProt", "identity", "alignment_score"
    ])
    df.to_csv(os.path.join(OUTPUT_DIR, "PDB_Binding_Residues_UniProt.csv"), index=False)
    return df

def union_binding_residues_by_gpcr(mapped_df):
    """Generate GPCR_Binding_Residues_UniProt.csv."""
    mapped_df["res_set"] = mapped_df["binding_res_uniprot"].apply(
        lambda x: set(ast.literal_eval(x)) if isinstance(x, str) else set(x)
    )
    records = []
    for uni_ac, grp in mapped_df.groupby("UniProt_ID"):
        union_set = set().union(*grp["res_set"])
        records.append([uni_ac, sorted(list(union_set)), len(union_set)])
    df = pd.DataFrame(records, columns=["UniProt_ID", "Binding_Residues", "Residue_Count"])
    df.to_csv(os.path.join(OUTPUT_DIR, "GPCR_Binding_Residues_UniProt.csv"), index=False)
    return df

def process_gpcr_structures(df_pdb_info, dist_thresholds=(4.0, 6.0, 8.0)):
    """Main pipeline to process GPCR structures."""
    print("Processing representative chains...")
    df_chains = process_representative_chains(df_pdb_info)
    
    print("Processing ligands...")
    df_ligands = process_ligands(df_pdb_info)
    
    print("Processing binding residues...")
    df_binding_res = process_binding_residues(df_chains, df_ligands, dist_thresholds)
    
    print("Mapping binding residues to UniProt...")
    df_mapped_res = map_binding_residues_to_uniprot(df_binding_res)
    
    print("Generating union of binding residues per GPCR...")
    df_union_res = union_binding_residues_by_gpcr(df_mapped_res)
    
    # Process AlphaFold2 structures
    results_af2 = []
    for _, row in df_pdb_info.iterrows():
        uniprot_ac = str(row["Entry"]).strip()
        uni_seq = fetch_uniprot_sequence(uniprot_ac)
        af2_path = download_alphafold_structure(uniprot_ac)
        if not af2_path or not uni_seq:
            continue
        af2_structure = parse_structure(uniprot_ac, af2_path, "pdb")
        if not af2_structure:
            continue
        chain = next(af2_structure.get_models())['A']
        polypep = Polypeptide.Polypeptide(chain)
        af2_seq = str(polypep.get_sequence())
        aln = local_align_smith_waterman(af2_seq, uni_seq)
        af2_res_list = [r for r in chain if is_aa(r, standard=True)]
        res_map = map_pdb_res_to_uniprot(af2_res_list, aln["aligned_pdb_seq"], aln["aligned_uni_seq"])
        # Map binding residues from union set
        binding_row = df_union_res[df_union_res["UniProt_ID"] == uniprot_ac]
        if not binding_row.empty:
            binding_res = ast.literal_eval(binding_row.iloc[0]["Binding_Residues"])
            af2_binding = [r for r, idxu in res_map.items() if idxu in binding_res]
            results_af2.append({
                "UNIPROT_AC": uniprot_ac, "AF2_Binding_Residues": [r.id[1] for r in af2_binding]
            })
    
    df_af2 = pd.DataFrame(results_af2)
    df_ligands = pd.merge(df_ligands, df_pdb_info[["Entry", "Entry Name"]], how="left",
                          left_on="UNIPROT_AC", right_on="Entry")
    df_ligands = df_ligands[[
        "UNIPROT_AC", "Entry Name", "PDB_ID", "chain_id", "ligand_id",
        "resseq", "heavy_atom_count", "smiles", "inchi_key"
    ]]
    
    # Save outputs
    df_ligands.to_csv(os.path.join(OUTPUT_DIR, "GPCR_PDB_Ligand_Info.csv"), index=False)
    df_af2.to_csv(os.path.join(OUTPUT_DIR, "GPCR_AF2_Binding_Res.csv"), index=False)
    return df_ligands, df_chains, df_binding_res, df_mapped_res, df_union_res, df_af2

def main():
    """Execute the GPCR structure processing pipeline."""
    print("Starting GPCR structure processing at 03:58 PM KST, May 28, 2025...")
    df_pdb_info = pd.read_csv("./Input/Human_GPCR_PDB_Info.csv")
    df_ligands, df_chains, df_binding_res, df_mapped_res, df_union_res, df_af2 = process_gpcr_structures(df_pdb_info)
    print("Pipeline completed. Output files saved in ./Output/Binding_Residue/")
    print(f"- GPCR_PDB_Ligand_Info.csv: {len(df_ligands)} entries")
    print(f"- Rep_GPCR_chain.csv: {len(df_chains)} entries")
    print(f"- PDB_Binding_Res.csv: {len(df_binding_res)} entries")
    print(f"- PDB_Binding_Residues_UniProt.csv: {len(df_mapped_res)} entries")
    print(f"- GPCR_Binding_Residues_UniProt.csv: {len(df_union_res)} entries")
    print(f"- GPCR_AF2_Binding_Res.csv: {len(df_af2)} entries")

if __name__ == "__main__":
    main()
