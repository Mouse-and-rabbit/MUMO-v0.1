"""
MUMO Agent — Interaction Analyst
Multi-Agent Drug Discovery & Development AI Platform
Author: Mowriss & Claude (research partner)

WHAT THIS AGENT DOES (plain English)
------------------------------------
AutoDock Vina only tells us HOW STRONG the binding is (the affinity number).
It does NOT tell us WHY. This agent answers the "why":
    - How many total interactions hold the drug in place?
    - How many hydrogen bonds (the strongest, most specific 'handshakes')?
    - WHICH protein residues form those H-bonds?
    - How many hydrophobic contacts, pi-stacking, salt bridges, halogen bonds?

It does this with PLIP (Protein-Ligand Interaction Profiler) — the tool used in
published papers — by analysing the docked 3D pose against the protein.

HOW IT WORKS
    1. Take the best docked pose (from Vina's output) + the protein structure
    2. Glue them into one "complex" PDB file (protein + ligand together)
    3. Run PLIP on that complex
    4. Summarise every interaction into a clean dictionary
"""

import os
import openbabel.pybel as pybel

# ── Compatibility shim ───────────────────────────────────────────────────────
# The prebuilt OpenBabel wheel on Windows has no InChI support, but PLIP asks
# OpenBabel for an 'inchikey' (just to label the ligand). RDKit *does* have
# InChI, so we redirect only those two formats to RDKit. Everything else is
# untouched. This keeps PLIP fully working without changing PLIP itself.
_orig_write = pybel.Molecule.write
def _write_with_rdkit_inchi(self, format="smi", filename=None, *args, **kwargs):
    if format in ("inchikey", "inchi") and filename is None:
        try:
            from rdkit import Chem
            molblock = _orig_write(self, "mol")
            m = Chem.MolFromMolBlock(molblock, sanitize=False)
            if m is not None:
                return Chem.MolToInchiKey(m) if format == "inchikey" else Chem.MolToInchi(m)
        except Exception:
            pass
        return "NOINCHIKEY"   # harmless fallback label; PLIP still works
    return _orig_write(self, format, filename, *args, **kwargs)
pybel.Molecule.write = _write_with_rdkit_inchi

from plip.structure.preparation import PDBComplex


def _ligand_pose_to_pdb_block(ligand_pdbqt):
    """
    Take the FIRST (best) pose from Vina's output .pdbqt and turn it into PDB
    lines, marked as HETATM residue 'LIG' on chain Z so PLIP treats it as the
    ligand (not part of the protein).
    """
    mol = next(pybel.readfile("pdbqt", ligand_pdbqt))   # first pose = best pose
    pdb_text = mol.write("pdb")

    fixed = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            line = "HETATM" + line[6:]            # force HETATM
            line = line[:17] + "LIG" + line[20:]  # residue name -> LIG
            line = line[:21] + "Z" + line[22:]    # chain -> Z
            fixed.append(line)
    return "\n".join(fixed)


def build_complex(receptor_pdb, ligand_pdbqt, out_complex_pdb):
    """Glue protein + best ligand pose into one complex PDB file for PLIP."""
    with open(receptor_pdb) as f:
        protein_lines = [ln.rstrip("\n") for ln in f
                         if ln.startswith(("ATOM", "TER"))]
    ligand_block = _ligand_pose_to_pdb_block(ligand_pdbqt)

    with open(out_complex_pdb, "w") as f:
        f.write("\n".join(protein_lines) + "\n")
        f.write("TER\n")
        f.write(ligand_block + "\n")
        f.write("END\n")
    return out_complex_pdb


def _residue_tag(i):
    """Make a readable residue label like 'ASP110(A)' from a PLIP interaction."""
    return f"{i.restype}{i.resnr}({i.reschain})"


def _xyz(obj):
    """Get an (x,y,z) tuple from a pybel atom (.coords) or a ring center (.center)."""
    if hasattr(obj, "coords"):
        return tuple(obj.coords)
    return tuple(obj)   # already a coordinate list/array


def _extract_lines(site):
    """
    Build a list of interaction 'lines' to draw in 3D:
        {type, p1:(x,y,z), p2:(x,y,z), color, label}
    Each line connects the two atoms/centres involved in an interaction.
    Wrapped in try/except per item so one odd record never breaks the picture.
    """
    lines = []
    def add(kind, a, b, color, label):
        try:
            lines.append({"type": kind, "p1": _xyz(a), "p2": _xyz(b),
                          "color": color, "label": label})
        except Exception:
            pass

    for h in list(site.hbonds_ldon) + list(site.hbonds_pdon):
        add("H-bond", h.a, h.d, "blue", _residue_tag(h))
    for c in site.hydrophobic_contacts:
        add("Hydrophobic", c.ligatom, c.bsatom, "grey", _residue_tag(c))
    for p in site.pistacking:
        add("Pi-stacking", p.ligandring.center, p.proteinring.center, "green", _residue_tag(p))
    for s in list(site.saltbridge_lneg) + list(site.saltbridge_pneg):
        add("Salt bridge", s.positive.center, s.negative.center, "orange", _residue_tag(s))
    for x in site.halogen_bonds:
        add("Halogen", x.don.x, x.acc.o, "cyan", _residue_tag(x))
    return lines


def analyze_interactions(receptor_pdb, ligand_pdbqt, out_complex_pdb):
    """
    Full analysis. Returns a dictionary of interaction details for the docked pose.
    """
    build_complex(receptor_pdb, ligand_pdbqt, out_complex_pdb)

    complex_mol = PDBComplex()
    complex_mol.load_pdb(out_complex_pdb)
    complex_mol.analyze()

    if not complex_mol.interaction_sets:
        raise RuntimeError("PLIP found no ligand binding site in the complex.")

    # Pick the binding site with the most interactions (our docked ligand).
    def total(site):
        return (len(site.hbonds_ldon) + len(site.hbonds_pdon) +
                len(site.hydrophobic_contacts) + len(site.pistacking) +
                len(site.saltbridge_lneg) + len(site.saltbridge_pneg) +
                len(site.halogen_bonds) + len(site.pication_laro) +
                len(site.pication_paro) + len(site.water_bridges))

    site = max(complex_mol.interaction_sets.values(), key=total)

    hbonds = list(site.hbonds_ldon) + list(site.hbonds_pdon)
    hydrophobic = list(site.hydrophobic_contacts)
    pistacking = list(site.pistacking)
    saltbridges = list(site.saltbridge_lneg) + list(site.saltbridge_pneg)
    halogens = list(site.halogen_bonds)
    pication = list(site.pication_laro) + list(site.pication_paro)
    waterbridges = list(site.water_bridges)

    all_residues = sorted({_residue_tag(i) for group in
                           (hbonds, hydrophobic, pistacking, saltbridges, halogens, pication)
                           for i in group})

    return {
        "total_interactions": (len(hbonds) + len(hydrophobic) + len(pistacking) +
                               len(saltbridges) + len(halogens) + len(pication) +
                               len(waterbridges)),
        "n_hbonds": len(hbonds),
        "hbond_residues": [_residue_tag(i) for i in hbonds],
        "n_hydrophobic": len(hydrophobic),
        "hydrophobic_residues": [_residue_tag(i) for i in hydrophobic],
        "n_pistacking": len(pistacking),
        "pistacking_residues": [_residue_tag(i) for i in pistacking],
        "n_saltbridges": len(saltbridges),
        "saltbridge_residues": [_residue_tag(i) for i in saltbridges],
        "n_halogen": len(halogens),
        "n_pication": len(pication),
        "n_waterbridges": len(waterbridges),
        "interacting_residues": all_residues,
        "lines": _extract_lines(site),       # 3D coords for drawing interactions
        "residue_numbers": sorted({i.resnr for group in
                                   (hbonds, hydrophobic, pistacking, saltbridges, halogens, pication)
                                   for i in group}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — analyse the CFTR complex we already docked earlier
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA = os.path.join(PROJECT, "data")
    receptor = os.path.join(DATA, "chain_cleaned.pdb")
    ligand   = os.path.join(DATA, "chain_out.pdbqt")
    complexf = os.path.join(DATA, "chain_complex.pdb")

    print("=" * 60)
    print("MUMO INTERACTION ANALYST — CFTR docked complex")
    print("=" * 60)
    res = analyze_interactions(receptor, ligand, complexf)
    print(f"Total interactions : {res['total_interactions']}")
    print(f"Hydrogen bonds     : {res['n_hbonds']}  -> {res['hbond_residues']}")
    print(f"Hydrophobic        : {res['n_hydrophobic']} -> {res['hydrophobic_residues']}")
    print(f"Pi-stacking        : {res['n_pistacking']} -> {res['pistacking_residues']}")
    print(f"Salt bridges       : {res['n_saltbridges']} -> {res['saltbridge_residues']}")
    print(f"Halogen bonds      : {res['n_halogen']}")
    print(f"Water bridges      : {res['n_waterbridges']}")
    print(f"All residues       : {res['interacting_residues']}")
