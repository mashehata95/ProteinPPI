"""
ProteinPPI — modular protein-protein interaction analysis package.

Sub-modules
-----------
structure              StructureLoader      — file parsing, chain/residue helpers
sasa                   SASACalculator       — SASA, BSA, ΔSASA per residue
interactions           InteractionFinder    — 10 interaction types + ΔSASA columns
shape_complementarity  ShapeComplementarity — SC score normalised to [0, 1]
affinity               ProdigyCalculator    — binding affinity (PRODIGY) +
                                             pKa per residue (PROPKA3) +
                                             pH-dependent protonation (PDB2PQR)
"""

from .structure import StructureLoader
from .sasa import SASACalculator
from .interactions import InteractionFinder
from .shape_complementarity import ShapeComplementarity
from .affinity import ProdigyCalculator

__all__ = [
    "StructureLoader",
    "SASACalculator",
    "InteractionFinder",
    "ShapeComplementarity",
    "ProdigyCalculator"
]
