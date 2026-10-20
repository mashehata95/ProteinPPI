"""
affinity.py — ProdigyCalculator
=================================
Computes protein-protein binding affinity using PRODIGY and pKa values
using PROPKA3, with optional structure protonation via PDB2PQR at a given pH.

Requirements
------------
Install via pip::

    pip install propka pdb2pqr prodigy-prot

Algorithm (per chain pair)
--------------------------
1.  Write a protein-only PDB file for the A+B complex.
2.  Run **PROPKA3** to predict pKa values for all ionisable residues.
3.  Run **PDB2PQR** (using PROPKA) to protonate the structure at the given pH.
4.  Convert the PQR output to PDB format.
5.  Run **PRODIGY** on the protonated structure to predict ΔG (kcal/mol) and Kd (M).

pKa output columns (pka_<AB>.csv)
-----------------------------------
chain, residue, resnum, pka, model_pka

Affinity output (added to interface_metrics.csv)
-------------------------------------------------
delta_g_kcal_mol, kd_M

Notes
-----
*  pKa values are predicted for ionisable residues only (ASP, GLU, HIS, LYS,
   ARG, TYR, CYS, N-/C-termini).  Non-ionisable residues have empty pka fields
   in the interaction CSV.
*  If PDB2PQR fails the original (unprotonated) PDB is passed to PRODIGY with
   a warning.  PRODIGY's contact-counting algorithm is robust to missing
   explicit hydrogens.
*  PRODIGY uses intermolecular contacts and charged-residue classification;
   the pH-dependent protonation state influences which residues are classified
   as charged or neutral.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

from Bio.PDB import PDBIO
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.Structure import Structure as PDBStructure

from .structure import StructureLoader


_PROPKA  = "propka3"
_PDB2PQR = "pdb2pqr"
_PRODIGY = "prodigy"


class ProdigyCalculator:
    """Predict binding affinity (PRODIGY) and pKa values (PROPKA3) for a chain pair.

    Parameters
    ----------
    loader:
        A fully initialised :class:`~proteinppi.structure.StructureLoader`.
    pH:
        pH used for structure protonation via PDB2PQR (default 7.4).
    temperature:
        Temperature (°C) passed to PRODIGY for Kd calculation (default 25.0).
    """

    PKA_FIELDNAMES: list[str] = [
        "chain", "residue", "resnum", "pka", "model_pka",
    ]
    AFFINITY_FIELDNAMES: list[str] = ["delta_g_kcal_mol", "kd_M"]

    def __init__(
        self,
        loader:      StructureLoader,
        pH:          float = 7.4,
        temperature: float = 25.0,
    ) -> None:
        self.loader      = loader
        self.pH          = pH
        self.temperature = temperature
        self._check_dependencies()

    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_dependencies() -> None:
        """Raise RuntimeError if any required executable is missing from PATH."""
        missing = [
            exe for exe in (_PROPKA, _PDB2PQR, _PRODIGY)
            if shutil.which(exe) is None
        ]
        if missing:
            raise RuntimeError(
                "Required executables not found on PATH: "
                + ", ".join(missing)
                + "\n\nInstall with:\n"
                "    pip install propka pdb2pqr prodigy-prot"
            )

    # ------------------------------------------------------------------
    # PDB writer (protein-only, HETATM stripped)
    # ------------------------------------------------------------------

    def _write_chain_pdb(self, chain_ids: list[str], out_path: str) -> str:
        """Write selected protein-only chains to *out_path* in PDB format."""
        new_struct = PDBStructure("tmp")
        new_model  = PDBModel(0)
        for chain in self.loader.structure.get_chains():
            if chain.id in chain_ids:
                new_model.add(chain.copy())
        new_struct.add(new_model)

        for model in new_struct:
            for chain in model:
                to_remove = [
                    r.id for r in chain.get_residues()
                    if r.get_id()[0] != " "
                    or r.resname.strip() not in StructureLoader.STANDARD_AA
                ]
                for rid in to_remove:
                    chain.detach_child(rid)

        io = PDBIO()
        io.set_structure(new_struct)
        io.save(out_path)
        return out_path

    # ------------------------------------------------------------------
    # PROPKA3: pKa prediction
    # ------------------------------------------------------------------

    def _run_propka(
        self,
        pdb_path: str,
        tmpdir:   str,
    ) -> dict[tuple[str, int], dict]:
        """Run PROPKA3 on *pdb_path* and return a pKa lookup dict.

        Returns
        -------
        dict mapping ``(chain_id, resnum)`` to
        ``{"resname": str, "pka": float, "model_pka": float}``.
        """
        result = subprocess.run(
            [_PROPKA, pdb_path],
            capture_output=True, text=True, cwd=tmpdir,
        )

        # PROPKA3 writes the .pka file alongside the input; some versions use CWD
        stem     = os.path.splitext(os.path.basename(pdb_path))[0]
        pka_same = os.path.splitext(pdb_path)[0] + ".pka"
        pka_cwd  = os.path.join(tmpdir, stem + ".pka")
        pka_file = pka_same if os.path.exists(pka_same) else (
            pka_cwd if os.path.exists(pka_cwd) else None
        )

        if pka_file is None:
            raise RuntimeError(
                f"PROPKA3 did not produce a .pka file (rc={result.returncode}).\n"
                f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
            )

        return self._parse_pka_file(pka_file)

    @staticmethod
    def _parse_pka_file(pka_file: str) -> dict[tuple[str, int], dict]:
        """Parse the SUMMARY section of a PROPKA3 .pka file.

        Handles standard residue entries and terminal groups (N+, C-).

        Returns
        -------
        dict mapping ``(chain, resnum)`` → ``{"resname", "pka", "model_pka"}``.
        """
        pka_dict: dict[tuple[str, int], dict] = {}

        # Match summary lines:
        #    ASP   6 A                2.00       3.80
        #    HIS  50 A                6.38       6.50
        #    N+    1 A                8.33       8.00
        pat = re.compile(
            r"^\s*(\S+)"          # resname (may contain +/-)
            r"\s+(\d+)"           # resnum
            r"\s+([A-Za-z\d])"   # chain ID
            r"\s+([\d.]+)"        # pKa
            r"\s+([\d.]+)",       # model-pKa
        )
        in_summary = False
        with open(pka_file) as fh:
            for line in fh:
                if "SUMMARY OF THIS PREDICTION" in line:
                    in_summary = True
                    continue
                if not in_summary:
                    continue
                m = pat.match(line)
                if m:
                    resname   = m.group(1)
                    resnum    = int(m.group(2))
                    chain     = m.group(3)
                    pka_val   = float(m.group(4))
                    model_pka = float(m.group(5))
                    pka_dict[(chain, resnum)] = {
                        "resname":   resname,
                        "pka":       pka_val,
                        "model_pka": model_pka,
                    }

        return pka_dict

    def pka_records(
        self,
        pka_dict:  dict[tuple[str, int], dict],
        chain_ids: list[str],
    ) -> list[dict]:
        """Return a list of pKa records (for CSV output) filtered to *chain_ids*."""
        rows = []
        for (chain, resnum), info in sorted(
            pka_dict.items(), key=lambda kv: (kv[0][0], kv[0][1])
        ):
            if chain not in chain_ids:
                continue
            rows.append({
                "chain":     chain,
                "residue":   f"{info['resname']}{resnum}",
                "resnum":    resnum,
                "pka":       info["pka"],
                "model_pka": info["model_pka"],
            })
        return rows

    # ------------------------------------------------------------------
    # PDB2PQR: protonation at given pH
    # ------------------------------------------------------------------

    def _run_pdb2pqr(
        self,
        pdb_path:  str,
        out_pqr:   str,
        chain_ids: list[str],
        tmpdir:    str,
    ) -> str:
        """Protonate *pdb_path* at ``self.pH`` using PDB2PQR + PROPKA.

        Returns the path to a PDB-format file converted from the PQR output.
        Tries the PDB2PQR 3.x argument style first, then the 2.x style.
        """
        common = [
            "--ff=AMBER",
            f"--with-ph={self.pH:.2f}",
            "--quiet",
            pdb_path,
            out_pqr,
        ]
        # pdb2pqr 3.x uses --titration-state-method
        result = subprocess.run(
            [_PDB2PQR, "--titration-state-method=propka"] + common,
            capture_output=True, text=True, cwd=tmpdir,
        )

        if result.returncode != 0 or not os.path.exists(out_pqr):
            # pdb2pqr 2.x uses --ph-calc-method
            result = subprocess.run(
                [_PDB2PQR, "--ph-calc-method=propka"] + common,
                capture_output=True, text=True, cwd=tmpdir,
            )

        if not os.path.exists(out_pqr):
            raise RuntimeError(
                f"PDB2PQR failed (rc={result.returncode}).\n"
                f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
            )

        out_pdb = os.path.splitext(out_pqr)[0] + "_protonated.pdb"
        self._pqr_to_pdb(out_pqr, out_pdb, chain_ids)
        return out_pdb

    # Residue name normalisations applied after PDB2PQR protonation.
    # PDB2PQR renames residues to reflect protonation state; PRODIGY expects
    # standard 3-letter codes, so we map them back before writing the PDB.
    _PDB2PQR_RESNAME_MAP: dict[str, str] = {
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS",   # histidine tautomers
        "ASH": "ASP",                                  # protonated aspartate
        "GLH": "GLU",                                  # protonated glutamate
        "LYN": "LYS",                                  # neutral lysine
        "CYX": "CYS", "CYM": "CYS",                   # disulfide / deprotonated cys
    }

    def _pqr_to_pdb(
        self,
        pqr_path:  str,
        pdb_path:  str,
        chain_ids: list[str],
    ) -> None:
        """Convert a PQR file to PDB format for PRODIGY.

        Three issues are corrected here:

        1. **Blank chain IDs** — PDB2PQR 3.x strips chain IDs from the PQR.
           We restore them by matching the sequential residue order against
           the original loaded structure.
        2. **Fixed-width columns** — PDB/PQR uses fixed column positions
           (chain=21, resseq=22-25, x=30-37).  ``split()`` is wrong when
           the chain field is a blank space.
        3. **Hydrogen atoms** — PRODIGY/freesasa has no radii for many PDB2PQR-
           added hydrogens.  We drop all hydrogen atoms (elem='H').
           Residue names renamed by PDB2PQR (HID/HIE/HIP → HIS, etc.) are
           normalised back to standard codes so PRODIGY recognises them.
        """
        # Build ordered list of (chain, resnum) from the original structure
        # so we can reassign chain IDs to PQR atoms by sequential position.
        orig_residues: list[tuple[str, int]] = []
        for cid in chain_ids:
            for chain in self.loader.structure.get_chains():
                if chain.id == cid:
                    for res in chain.get_residues():
                        if (res.get_id()[0] == " "
                                and res.resname.strip() in StructureLoader.STANDARD_AA):
                            orig_residues.append((cid, res.get_id()[1]))

        pqr_res_idx  = -1    # index into orig_residues; -1 = not yet started
        last_res_key = None  # (resnum, normalised_resname) of last seen residue

        with open(pqr_path) as src, open(pdb_path, "w") as dst:
            for line in src:
                rec = line[:6].strip()
                if rec not in ("ATOM", "HETATM"):
                    continue
                if len(line) < 54:
                    continue

                # Fixed-width column parsing (PDB / PQR layout)
                try:
                    serial  = line[6:11].strip()
                    name    = line[12:16]          # 4 chars, keep spacing
                    resname = line[17:20].strip()
                    resseq  = line[22:26].strip()
                    x       = float(line[30:38])
                    y       = float(line[38:46])
                    z       = float(line[46:54])
                except (ValueError, IndexError):
                    continue

                if not resseq.lstrip("-").isdigit():
                    continue
                resnum = int(resseq)

                # Determine element from atom name; skip hydrogens
                elem = re.sub(r"[^A-Za-z]", "", name.strip())[:1]
                if elem == "H":
                    continue

                # Normalise PDB2PQR-specific residue names → standard codes
                std_resname = self._PDB2PQR_RESNAME_MAP.get(resname, resname)

                # Sequential chain-ID restoration: advance the residue index
                # each time the (resnum, resname) pair changes (hydrogens of
                # the same residue share the same pair, so they don't advance).
                curr_key = (resnum, std_resname)
                if curr_key != last_res_key:
                    if std_resname in StructureLoader.STANDARD_AA:
                        pqr_res_idx += 1
                    last_res_key = curr_key

                if 0 <= pqr_res_idx < len(orig_residues):
                    chain = orig_residues[pqr_res_idx][0]
                else:
                    chain = chain_ids[-1] if chain_ids else " "

                full_elem = re.sub(r"[^A-Za-z]", "", name.strip())[:2]

                # Write standard PDB fixed-width record
                # Positions: rec[0:6] serial[6:11] ' '[11] name[12:15]
                #            altloc=' '[16] resname[17:19] ' '[20]
                #            chain[21] resseq[22:25] icode+3sp[26:29]
                #            x[30:37] y[38:45] z[46:53]
                dst.write(
                    f"ATOM  {serial:>5} {name} {std_resname:<3} {chain}"
                    f"{resnum:>4}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {full_elem:>2}\n"
                )
            dst.write("END\n")

    # ------------------------------------------------------------------
    # PRODIGY: binding affinity
    # ------------------------------------------------------------------

    def _run_prodigy(
        self,
        pdb_path: str,
        chain_a:  str,
        chain_b:  str,
        tmpdir:   str,
    ) -> tuple[Optional[float], Optional[float]]:
        """Run PRODIGY on *pdb_path* and return ``(delta_g_kcal_mol, kd_M)``."""
        cmd = [
            _PRODIGY,
            "--selection", chain_a, chain_b,
            "--temperature", str(self.temperature),
            pdb_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=tmpdir,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"PRODIGY failed (rc={result.returncode}).\n"
                f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
            )

        output = result.stdout + result.stderr

        # ΔG: "Predicted binding affinity (kcal mol-1): -10.5"
        dg_m = re.search(
            r"(?:Predicted\s+binding\s+affinity|Predicted\s+ΔG)"
            r"[^:\n]*:\s*([-\d.]+)",
            output, re.IGNORECASE,
        )
        # Kd: "Predicted dissociation constant (M) at 25.0°C: 2.3e-08"
        kd_m = re.search(
            r"(?:Predicted\s+dissociation\s+constant|Predicted\s+Kd)"
            r"[^:\n]*:\s*([\d.eE+\-]+)",
            output, re.IGNORECASE,
        )

        delta_g = float(dg_m.group(1)) if dg_m else None
        kd      = float(kd_m.group(1)) if kd_m else None
        return delta_g, kd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_pka(self, chain_ids: list[str]) -> tuple[dict, list[dict]]:
        """Run PROPKA3 on the given chains and return pKa data.

        Parameters
        ----------
        chain_ids:
            Chain identifiers whose residues should be included.

        Returns
        -------
        pka_dict:
            Lookup dict ``{(chain, resnum): {"resname", "pka", "model_pka"}}``.
        records:
            List of dicts keyed by :attr:`PKA_FIELDNAMES` — ready for CSV output.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = os.path.join(tmpdir, "input.pdb")
            self._write_chain_pdb(chain_ids, pdb_path)
            pka_dict = self._run_propka(pdb_path, tmpdir)

        records = self.pka_records(pka_dict, chain_ids)
        return pka_dict, records

    def compute_affinity(
        self,
        chain_a_id: str,
        chain_b_id: str,
    ) -> dict:
        """Run PROPKA3 → PDB2PQR → PRODIGY for one chain pair.

        Parameters
        ----------
        chain_a_id, chain_b_id:
            Chain identifiers present in the loaded structure.

        Returns
        -------
        dict with keys:

        ``delta_g_kcal_mol`` : float | None
            Predicted binding free energy (kcal/mol).  Negative = favourable.
        ``kd_M`` : float | None
            Predicted dissociation constant (mol/L).
        ``pka_dict`` : dict
            See :meth:`compute_pka`.
        ``pka_records`` : list[dict]
            See :attr:`PKA_FIELDNAMES`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:

            # 1. Write complex PDB
            pdb_complex = os.path.join(tmpdir, "complex.pdb")
            self._write_chain_pdb([chain_a_id, chain_b_id], pdb_complex)
            print(f"    [affinity] PDB written ({chain_a_id}+{chain_b_id})")

            # 2. pKa prediction
            print("    [affinity] Running PROPKA3 …")
            pka_dict = self._run_propka(pdb_complex, tmpdir)
            records  = self.pka_records(pka_dict, [chain_a_id, chain_b_id])
            print(f"    [affinity] {len(records)} ionisable residues with pKa predicted.")

            # 3. Protonation at given pH
            print(f"    [affinity] Protonating at pH {self.pH:.2f} (PDB2PQR) …")
            out_pqr = os.path.join(tmpdir, "complex_protonated.pqr")
            try:
                prot_pdb = self._run_pdb2pqr(
                    pdb_complex, out_pqr, [chain_a_id, chain_b_id], tmpdir
                )
            except RuntimeError as exc:
                print(
                    f"    [affinity] WARNING: protonation failed — {exc}\n"
                    "    [affinity] Falling back to original (unprotonated) PDB."
                )
                prot_pdb = pdb_complex

            # 4. Binding affinity
            print(
                f"    [affinity] Running PRODIGY "
                f"(chains {chain_a_id},{chain_b_id}, "
                f"T={self.temperature}°C) …"
            )
            delta_g, kd = self._run_prodigy(
                prot_pdb, chain_a_id, chain_b_id, tmpdir
            )

        if delta_g is not None:
            print(
                f"    [affinity] ΔG = {delta_g:.3f} kcal/mol  "
                f"Kd = {kd:.3e} M"
            )
        else:
            print("    [affinity] WARNING: could not parse PRODIGY output.")

        return {
            "delta_g_kcal_mol": delta_g,
            "kd_M":             kd,
            "pka_dict":         pka_dict,
            "pka_records":      records,
        }
