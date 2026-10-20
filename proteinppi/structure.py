"""
structure.py — StructureLoader
==============================
Parses a PDB or mmCIF file, identifies protein chains (excluding glycans,
nucleic acids, and small molecules), and collects water molecules.

Exposes residue-level helpers used by every other module so they share a
single, consistent definition of "residue label" and "protein residue".
"""

from __future__ import annotations

import itertools
import os
from typing import Optional

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue


class StructureLoader:
    """Parse a structure file and expose protein chain / residue utilities.

    Parameters
    ----------
    structure_file:
        Path to a ``.pdb``, ``.ent``, ``.cif``, or ``.mmcif`` file.

    Raises
    ------
    ValueError
        If the file extension is unsupported or fewer than two protein
        chains are detected.

    Attributes
    ----------
    structure : Bio.PDB.Structure
        The parsed BioPython structure object (model 0 is always used).
    chain_ids : list[str]
        Sorted identifiers of detected protein chains.
    chain_pairs : list[tuple[str, str]]
        All pairwise combinations of ``chain_ids``.
    waters : list[Residue]
        All water residues found across the full structure.
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------

    _PARSERS: dict = {
        ".pdb":   lambda: PDBParser(QUIET=True),
        ".ent":   lambda: PDBParser(QUIET=True),
        ".cif":   lambda: MMCIFParser(QUIET=True),
        ".mmcif": lambda: MMCIFParser(QUIET=True),
    }

    STANDARD_AA: frozenset[str] = frozenset({
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    })

    ONE_LETTER: dict[str, str] = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    WATER_NAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD"})

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, structure_file: str) -> None:
        ext = os.path.splitext(structure_file)[1].lower()
        parser_factory = self._PARSERS.get(ext)
        if parser_factory is None:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Supported extensions: {sorted(self._PARSERS)}"
            )

        self.structure = parser_factory().get_structure("structure", structure_file)

        self.chain_ids: list[str] = [
            c.id for c in self.structure.get_chains()
            if self._is_protein_chain(c)
        ]
        if len(self.chain_ids) < 2:
            raise ValueError(
                f"At least 2 protein chains are required; "
                f"found: {self.chain_ids or 'none'}"
            )

        self.chain_pairs: list[tuple[str, str]] = list(
            itertools.combinations(self.chain_ids, 2)
        )

        self.waters: list[Residue] = [
            r for r in self.structure.get_residues()
            if r.resname in self.WATER_NAMES
        ]

        print(f"Protein chains  : {self.chain_ids}")
        print(f"Chain pairs     : {[f'{a}-{b}' for a, b in self.chain_pairs]}")
        print(f"Water molecules : {len(self.waters)}")

    # ------------------------------------------------------------------
    # Chain / residue helpers  (used by every other module)
    # ------------------------------------------------------------------

    def _is_protein_chain(self, chain: Chain) -> bool:
        """Return True if *chain* contains ≥1 standard amino-acid residue."""
        return any(
            r.get_id()[0] == " " and r.resname.strip() in self.STANDARD_AA
            for r in chain.get_residues()
        )

    def protein_residues(self, chain_id: str) -> list[Residue]:
        """Return all standard amino-acid residues for *chain_id* (model 0)."""
        return [
            r for r in self.structure[0][chain_id].get_residues()
            if r.get_id()[0] == " " and r.resname in self.STANDARD_AA
        ]

    def res_label(self, residue: Residue) -> str:
        """Compact residue label: one-letter code + sequence number (e.g. ``K23``)."""
        letter = self.ONE_LETTER.get(residue.resname, "X")
        return letter + str(residue.get_id()[1])

    def ca_coord(self, residue: Residue) -> Optional[np.ndarray]:
        """Return the Cα coordinate of *residue*, or ``None`` if absent."""
        try:
            return residue["CA"].get_coord()
        except KeyError:
            return None

    @staticmethod
    def dist(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two coordinate arrays."""
        return float(np.linalg.norm(a - b))
