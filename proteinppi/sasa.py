"""
sasa.py — SASACalculator
========================
Computes Solvent-Accessible Surface Area (SASA) values for protein chain
pairs using **freesasa** (Lee-Richards algorithm, 1.4 Å water probe).

Responsibilities
----------------
* Write temporary PDB files for individual chains and their complex.
* Compute total SASA for each chain and the complex.
* Compute Buried Surface Area (BSA):
      BSA = (SASA_A + SASA_B − SASA_AB) / 2
* Compute per-residue ΔSASA (monomer minus complex) for both chains,
  which is consumed by InteractionFinder to annotate every interaction row.
* Identify interface residues (ΔSASA > 0) for use by ShapeComplementarity.

All temporary files are created inside a ``tempfile.TemporaryDirectory``
that is cleaned up automatically after each ``compute_for_pair`` call.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import freesasa
from Bio.PDB import PDBIO
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure as PDBStructure

from .structure import StructureLoader


class SASACalculator:
    """Compute SASA-based metrics for a pair of protein chains.

    Parameters
    ----------
    loader:
        A fully initialised :class:`StructureLoader`.
    """

    def __init__(self, loader: StructureLoader) -> None:
        self.loader = loader

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_chain_pdb(self, chain_ids: list[str], out_path: str) -> str:
        """Write selected chains to *out_path* as a plain PDB file.

        HETATM records (glycans, ligands, waters) are stripped so freesasa
        only sees protein atoms.  Returns *out_path* for convenience.
        """
        new_structure = PDBStructure("tmp")
        new_model = PDBModel(0)
        for chain in self.loader.structure.get_chains():
            if chain.id in chain_ids:
                new_model.add(chain.copy())
        new_structure.add(new_model)

        for model in new_structure:
            for chain in model:
                to_remove = [
                    r.id for r in chain.get_residues()
                    if r.get_id()[0] != " "
                ]
                for rid in to_remove:
                    chain.detach_child(rid)

        io = PDBIO()
        io.set_structure(new_structure)
        io.save(out_path)
        return out_path

    @staticmethod
    def _freesasa_total(pdb_path: str) -> float:
        """Return the total SASA (Å²) for the structure in *pdb_path*."""
        return freesasa.calc(freesasa.Structure(pdb_path)).totalArea()

    def _dsasa_per_residue(
        self,
        chain_id: str,
        path_mono: str,
        path_complex: str,
    ) -> dict[str, float]:
        """Return ΔSASA per residue label for *chain_id*.

        ΔSASA(residue) = SASA_monomer(residue) − SASA_complex(residue)

        A positive value means the residue loses surface area upon complex
        formation (i.e. it is at the interface).

        Keys are residue labels produced by :meth:`StructureLoader.res_label`
        (e.g. ``"K23"``), so they match the interaction CSV residue columns.
        Residues with ΔSASA ≤ 0 are omitted from the dict.
        """
        result_mono    = freesasa.calc(freesasa.Structure(path_mono))
        result_complex = freesasa.calc(freesasa.Structure(path_complex))

        mono_areas    = result_mono.residueAreas()
        complex_areas = result_complex.residueAreas()

        # freesasa may relabel the single chain in a monomer file
        mono_key    = chain_id if chain_id in mono_areas    else next(iter(mono_areas),    None)
        complex_key = chain_id if chain_id in complex_areas else None

        if mono_key is None or complex_key is None:
            return {}

        dsasa: dict[str, float] = {}
        for res_num, mono_area in mono_areas[mono_key].items():
            complex_area = complex_areas[complex_key].get(res_num)
            if complex_area is None:
                continue
            delta = mono_area.total - complex_area.total
            if delta > 0.0:
                # Build a label matching StructureLoader.res_label()
                one_letter = StructureLoader.ONE_LETTER.get(
                    mono_area.residueType, "X"
                )
                label = one_letter + res_num.strip()
                dsasa[label] = round(delta, 3)

        return dsasa

    def _interface_residues(
        self,
        chain_id: str,
        dsasa: dict[str, float],
    ) -> list[Residue]:
        """Return protein residues whose label appears in *dsasa* (ΔSASA > 0)."""
        labels_with_delta = set(dsasa.keys())
        return [
            r for r in self.loader.protein_residues(chain_id)
            if self.loader.res_label(r) in labels_with_delta
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_for_pair(
        self, chain_a_id: str, chain_b_id: str
    ) -> dict:
        """Compute all SASA-based metrics for a chain pair.

        Temporary PDB files are written, used, and automatically deleted
        within this call.

        Parameters
        ----------
        chain_a_id, chain_b_id:
            Chain identifiers that must exist in the loaded structure.

        Returns
        -------
        dict with keys:

        ``sasa_chain_a`` : float
            Total SASA of chain A as a monomer (Å²).
        ``sasa_chain_b`` : float
            Total SASA of chain B as a monomer (Å²).
        ``sasa_complex`` : float
            Total SASA of the A+B complex (Å²).
        ``bsa`` : float
            Buried Surface Area (Å²).
        ``dsasa_a`` : dict[str, float]
            Per-residue ΔSASA for chain A (residue label → Å²).
        ``dsasa_b`` : dict[str, float]
            Per-residue ΔSASA for chain B.
        ``interface_residues_a`` : list[Residue]
            BioPython Residue objects with ΔSASA > 0 in chain A.
        ``interface_residues_b`` : list[Residue]
            BioPython Residue objects with ΔSASA > 0 in chain B.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a  = self._write_chain_pdb(
                [chain_a_id],
                os.path.join(tmpdir, f"chain_{chain_a_id}.pdb"),
            )
            path_b  = self._write_chain_pdb(
                [chain_b_id],
                os.path.join(tmpdir, f"chain_{chain_b_id}.pdb"),
            )
            path_ab = self._write_chain_pdb(
                [chain_a_id, chain_b_id],
                os.path.join(tmpdir, f"complex_{chain_a_id}{chain_b_id}.pdb"),
            )

            sasa_a  = self._freesasa_total(path_a)
            sasa_b  = self._freesasa_total(path_b)
            sasa_ab = self._freesasa_total(path_ab)
            bsa     = (sasa_a + sasa_b - sasa_ab) / 2.0

            dsasa_a = self._dsasa_per_residue(chain_a_id, path_a,  path_ab)
            dsasa_b = self._dsasa_per_residue(chain_b_id, path_b,  path_ab)

        int_res_a = self._interface_residues(chain_a_id, dsasa_a)
        int_res_b = self._interface_residues(chain_b_id, dsasa_b)

        return {
            "sasa_chain_a":         round(sasa_a,  2),
            "sasa_chain_b":         round(sasa_b,  2),
            "sasa_complex":         round(sasa_ab, 2),
            "bsa":                  round(bsa,     2),
            "dsasa_a":              dsasa_a,
            "dsasa_b":              dsasa_b,
            "interface_residues_a": int_res_a,
            "interface_residues_b": int_res_b,
        }
