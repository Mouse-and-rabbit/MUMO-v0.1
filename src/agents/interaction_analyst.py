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

# ── Resilient imports ────────────────────────────────────────────────────────
# Interaction profiling needs OpenBabel + PLIP. If either is missing (e.g. a
# cloud build hiccup), MUMO must NOT crash — docking still works, we just skip
# the interaction details. INTERACTIONS_AVAILABLE tells the rest of the app.
INTERACTIONS_AVAILABLE = True
_IMPORT_ERROR = ""
try:
    import openbabel.pybel as pybel

    # Compatibility shim: some OpenBabel builds lack InChI, but PLIP asks for an
    # 'inchikey' (just a label). RDKit has InChI, so we redirect those calls.
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
            return "NOINCHIKEY"
        return _orig_write(self, format, filename, *args, **kwargs)
    pybel.Molecule.write = _write_with_rdkit_inchi

    from plip.structure.preparation import PDBComplex
except Exception as _e:                      # pragma: no cover
    INTERACTIONS_AVAILABLE = False
    _IMPORT_ERROR = f"{type(_e).__name__}: {_e}"


def _empty_result(note):
    """A zeroed interaction result so the app keeps working when PLIP is absent."""
    return {
        "total_interactions": 0, "n_hbonds": 0, "hbond_residues": [],
        "n_hydrophobic": 0, "hydrophobic_residues": [],
        "n_pistacking": 0, "pistacking_residues": [],
        "n_saltbridges": 0, "saltbridge_residues": [],
        "n_halogen": 0, "n_pication": 0, "n_waterbridges": 0,
        "interacting_residues": [], "lines": [], "residue_numbers": [],
        "note": note,
        "svg_2d": "",
    }


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
    Never raises — if PLIP is unavailable or analysis fails, returns zeros so the
    rest of MUMO (docking, scores, 3D view) keeps working.
    """
    if not INTERACTIONS_AVAILABLE:
        return _empty_result(f"Interaction profiling unavailable ({_IMPORT_ERROR}).")

    try:
        return _run_plip(receptor_pdb, ligand_pdbqt, out_complex_pdb)
    except Exception as e:
        return _empty_result(f"Interaction analysis skipped: {e}")


def _run_plip(receptor_pdb, ligand_pdbqt, out_complex_pdb):
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
        "svg_2d": generate_2d_interaction_svg(out_complex_pdb, site),
    }


def generate_2d_interaction_svg(complex_pdb_path, site):
    """
    Generates a 2D interaction diagram of the ligand showing color-coded
    highlighted atoms and residue labels for interactions from PLIP.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D

        # Read ligand lines from complex_pdb_path (chain Z, residue name LIG)
        lig_lines = []
        with open(complex_pdb_path) as f:
            for line in f:
                if "LIG" in line and (line.startswith("HETATM") or line.startswith("ATOM")):
                    lig_lines.append(line)
        if not lig_lines:
            return ""
        
        pdb_block = "".join(lig_lines)
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False)
        if mol is None:
            return ""

        # Map PDB serial numbers to RDKit indices
        pdb_to_rdkit = {}
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info:
                pdb_to_rdkit[info.GetSerialNumber()] = atom.GetIdx()

        atom_notes = {}
        atom_colors = {}
        highlight_atoms = []

        def add_interaction(serial, label, color_rgb):
            idx = pdb_to_rdkit.get(serial)
            if idx is not None:
                if idx not in atom_notes:
                    atom_notes[idx] = []
                atom_notes[idx].append(label)
                atom_colors[idx] = color_rgb
                if idx not in highlight_atoms:
                    highlight_atoms.append(idx)

        # 1. Hydrogen bonds (Donor-Acceptor)
        for b in list(site.hbonds_ldon) + list(site.hbonds_pdon):
            serial = b.a_orig_idx if b.protisdon else b.d_orig_idx
            res_label = f"{b.restype}{b.resnr}({b.reschain}) [H-bond]"
            add_interaction(serial, res_label, (0.18, 0.49, 0.86))  # Premium blue

        # 2. Hydrophobic contacts
        for c in site.hydrophobic_contacts:
            serial = c.ligatom_orig_idx
            res_label = f"{c.restype}{c.resnr}({c.reschain}) [Hydroph]"
            add_interaction(serial, res_label, (0.55, 0.55, 0.55))  # Dark grey

        # 3. Halogen bonds
        for x in site.halogen_bonds:
            serial = x.don_orig_idx if x.don_orig_idx in pdb_to_rdkit else x.acc_orig_idx
            res_label = f"{x.restype}{x.resnr}({x.reschain}) [Halogen]"
            add_interaction(serial, res_label, (0.0, 0.72, 0.72))  # Teal/Cyan

        # 4. Salt bridges
        for s in list(site.saltbridge_lneg) + list(site.saltbridge_pneg):
            res_label = f"{s.restype}{s.resnr}({s.reschain}) [Salt Bridge]"
            group = s.negative if s.protispos else s.positive
            if hasattr(group, "atoms_orig_idx"):
                for serial in group.atoms_orig_idx:
                    add_interaction(serial, res_label, (1.0, 0.5, 0.0))  # Orange

        # 5. Pi-stacking
        for p in site.pistacking:
            res_label = f"{p.restype}{p.resnr}({p.reschain}) [Pi-stack]"
            if hasattr(p.ligandring, "atoms_orig_idx"):
                for serial in p.ligandring.atoms_orig_idx:
                    add_interaction(serial, res_label, (0.0, 0.72, 0.2))  # Green

        # 6. Pi-cation
        for p in list(site.pication_laro) + list(site.pication_paro):
            res_label = f"{p.restype}{p.resnr}({p.reschain}) [Pi-cation]"
            if hasattr(p, "ring") and hasattr(p.ring, "atoms_orig_idx") and any(a in pdb_to_rdkit for a in p.ring.atoms_orig_idx):
                for serial in p.ring.atoms_orig_idx:
                    add_interaction(serial, res_label, (0.8, 0.2, 0.8))  # Purple
            elif hasattr(p, "charge") and hasattr(p.charge, "atoms_orig_idx"):
                for serial in p.charge.atoms_orig_idx:
                    add_interaction(serial, res_label, (0.8, 0.2, 0.8))  # Purple

        # Apply notes (labels) to RDKit mol
        for idx, notes in atom_notes.items():
            note_text = ", ".join(notes)
            mol.GetAtomWithIdx(idx).SetProp("atomNote", note_text)

        # Generate 2D depiction coordinates
        rdDepictor.Compute2DCoords(mol)

        # Render molecule to SVG format on a white card
        drawer = rdMolDraw2D.MolDraw2DSVG(550, 480)
        opts = drawer.drawOptions()
        opts.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
        opts.annotationFontScale = 0.82
        
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as e:
        print(f"Error generating 2D SVG: {e}")
        return ""


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
