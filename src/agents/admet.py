"""
MUMO Agent — ADMET / Drug-likeness (lite)
Multi-Agent Drug Discovery & Development AI Platform
Author: Mowriss & Claude (research partner)

WHAT THIS DOES (plain English)
------------------------------
Given a molecule (SMILES), compute its DRUG-LIKENESS — the quick physical-chemistry
profile medicinal chemists use to judge whether a molecule could become an oral drug:
    - Molecular weight, LogP (greasiness), H-bond donors/acceptors, polar surface area,
      rotatable bonds, QED (overall drug-likeness 0–1), and Lipinski Rule-of-5 violations.

This is the first piece of the ADMET Analyst. It also resolves a drug NAME
(e.g. "aspirin") into a SMILES via PubChem, so users can type names, not just SMILES.

All free — RDKit (local) + PubChem (public). No API key.
"""

import io
from urllib.parse import quote
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, QED


def parse_ligand_file(filename, data_bytes):
    """
    Read a ligand file and return [{label, smiles}, ...] for valid molecules.
    Supports:  .smi/.txt (SMILES [name] per line), .csv (a SMILES column), .sdf.
    """
    name = filename.lower()
    ligs = []

    if name.endswith(".sdf"):
        supplier = Chem.ForwardSDMolSupplier(io.BytesIO(data_bytes))
        for i, m in enumerate(supplier):
            if m is not None:
                lbl = m.GetProp("_Name") if m.HasProp("_Name") and m.GetProp("_Name") else f"mol_{i+1}"
                ligs.append({"label": lbl, "smiles": Chem.MolToSmiles(m)})

    elif name.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(io.BytesIO(data_bytes))
        scol = next((c for c in df.columns
                     if c.lower() in ("smiles", "smile", "canonical_smiles", "structure")),
                    df.columns[0])
        ncol = next((c for c in df.columns if c.lower() in ("name", "id", "label", "title", "compound")), None)
        for i, row in df.iterrows():
            ligs.append({"label": str(row[ncol]) if ncol else f"lig_{i+1}",
                         "smiles": str(row[scol]).strip()})

    else:  # .smi / .txt
        text = data_bytes.decode("utf-8", "ignore")
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line or line.lower().startswith("smiles"):
                continue
            parts = line.replace(",", " ").split()
            ligs.append({"label": parts[1] if len(parts) > 1 else f"lig_{i+1}",
                         "smiles": parts[0]})

    return [l for l in ligs if is_valid_smiles(l["smiles"])]


def is_valid_smiles(s):
    """True if the string parses as a real molecule."""
    return bool(s) and Chem.MolFromSmiles(s) is not None


def name_to_smiles(name):
    """Resolve a drug/compound NAME to a SMILES via PubChem. Returns None if not found."""
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{quote(name)}/property/SMILES,IsomericSMILES/JSON")
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            # PubChem renamed 'CanonicalSMILES' -> 'SMILES'; accept any available.
            return props.get("SMILES") or props.get("IsomericSMILES") or props.get("CanonicalSMILES")
    except Exception:
        pass
    return None


def druglikeness(smiles):
    """
    Compute the drug-likeness profile for one SMILES.
    Returns a dict of properties + Lipinski Rule-of-5 violation count + a verdict.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mw   = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    hbd  = Lipinski.NumHDonors(m)
    hba  = Lipinski.NumHAcceptors(m)
    tpsa = Descriptors.TPSA(m)
    rotb = Lipinski.NumRotatableBonds(m)
    qed  = QED.qed(m)

    # Lipinski Rule of 5: MW<500, LogP<5, HBD<=5, HBA<=10
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    verdict = ("🟢 drug-like" if violations == 0 else
               "🟡 1 RO5 violation" if violations == 1 else
               f"🔴 {violations} RO5 violations")

    return {
        "MW (g/mol)": round(mw, 1),
        "LogP": round(logp, 2),
        "H-bond donors": hbd,
        "H-bond acceptors": hba,
        "TPSA (Å²)": round(tpsa, 1),
        "Rotatable bonds": rotb,
        "QED (0–1)": round(qed, 3),
        "Lipinski violations": violations,
        "Verdict": verdict,
    }


def resolve_ligand(text):
    """
    Turn whatever the user gave (a SMILES or a drug name) into a valid SMILES.
    Returns (smiles, label) or (None, None) if it can't be resolved.
    """
    if is_valid_smiles(text):
        return text, "your molecule"
    smi = name_to_smiles(text)
    if smi and is_valid_smiles(smi):
        return smi, text          # keep the friendly name as the label
    return None, None


if __name__ == "__main__":
    for q in ["aspirin", "CC(=O)Nc1ccc(O)cc1", "ibuprofen"]:
        smi, label = resolve_ligand(q)
        print(f"\n{q}  ->  {smi}  ({label})")
        if smi:
            for k, v in druglikeness(smi).items():
                print(f"   {k}: {v}")
