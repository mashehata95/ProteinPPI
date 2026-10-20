"""
shape_complementarity.py — ShapeComplementarity
================================================
Computes the Shape Complementarity (SC) score between two protein chain
interfaces following the method of Lawrence & Colman (1993).

Algorithm summary
-----------------
1. **Surface dots** — for each interface atom a Fibonacci-sphere lattice of
   ``n_sphere_dots`` candidate points is placed at distance r_vdw + r_probe
   from the atom centre.  Points occluded by other atoms in the same chain
   (Lee-Richards criterion: dist < r_neighbour + r_probe) are discarded.
   Surface dots are generated in the *monomer* context (each chain alone) so
   the interface face is not blocked by the partner chain.

2. **Pairing** — for each source dot the nearest target dot is found via a
   ``cKDTree``.  Pairs within ``sc_dmax`` Å are included.

3. **Dot-product** — ``n_src · (−n_tgt)`` is computed for each pair.
   The sign flip reflects the fact that outward normals on facing surfaces
   point in opposite directions; +1 = perfect complementarity.

4. **SC raw** = median of all pair dot-products  (bidirectional average).

5. **Normalisation** to [0, 1]::

       sc_normalised = clip((sc_raw + 1) / 2, 0, 1)

   Mapping: −1 → 0.0 (anti-complementary), 0 → 0.5 (random), +1 → 1.0.

Typical values (normalised)
---------------------------
Antibody–antigen interface   : ~0.82
Enzyme–inhibitor interface   : ~0.85–0.88
Crystal contacts             : ~0.75
Random / non-specific        : ~0.50
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from scipy.spatial import cKDTree

from .structure import StructureLoader

# Type alias
_Dot = tuple[np.ndarray, np.ndarray]   # (coord_3, outward_normal_3)


class ShapeComplementarity:
    """Compute the normalised SC score for a pair of interface residue sets.

    Parameters
    ----------
    loader:
        A fully initialised :class:`~proteinppi.structure.StructureLoader`.
    n_sphere_dots:
        Number of Fibonacci-sphere dots generated per atom (default 100).
        Higher values increase accuracy at the cost of runtime.
    sc_dmax:
        Maximum dot-to-dot distance (Å) for pairing source and target dots
        (default 4.0 Å).
    sc_min_pairs:
        Minimum number of matched dot pairs required to report a score.
        Returns ``None`` when the interface is too small (default 10).
    """

    # ------------------------------------------------------------------
    # Van der Waals radii (Å) by element symbol
    # ------------------------------------------------------------------
    _VDW_RADII: dict[str, float] = {
        "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
        "P": 1.80, "H": 1.20, "SE": 1.90,
        "FE": 1.47, "ZN": 1.39, "MG": 1.73, "CA": 1.97,
    }
    _DEFAULT_VDW: float = 1.70
    _PROBE_RADIUS: float = 1.4   # water probe (Å)

    def __init__(
        self,
        loader: StructureLoader,
        n_sphere_dots: int = 100,
        sc_dmax: float = 4.0,
        sc_min_pairs: int = 10,
    ) -> None:
        self.loader        = loader
        self.n_sphere_dots = n_sphere_dots
        self.sc_dmax       = sc_dmax
        self.sc_min_pairs  = sc_min_pairs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _atom_element(self, atom: Atom) -> str:
        """Return the upper-case element symbol, guessing from atom name if needed."""
        elem = (atom.element or "").strip().upper()
        if not elem:
            name = atom.get_name().strip().lstrip("0123456789")
            elem = name[0].upper() if name else "C"
        return elem

    def _atom_vdw(self, atom: Atom) -> float:
        """Return the Van der Waals radius (Å) for *atom*."""
        return self._VDW_RADII.get(self._atom_element(atom), self._DEFAULT_VDW)

    @staticmethod
    def _fibonacci_sphere(n: int) -> np.ndarray:
        """Generate *n* unit vectors uniformly distributed on a sphere.

        Uses the Fibonacci / golden-angle lattice.  Shape: ``(n, 3)``.
        """
        golden = (1.0 + np.sqrt(5.0)) / 2.0
        i      = np.arange(n, dtype=float)
        theta  = np.arccos(np.clip(1.0 - 2.0 * (i + 0.5) / n, -1.0, 1.0))
        phi    = 2.0 * np.pi * i / golden
        return np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])

    def _surface_dots(
        self,
        interface_residues: list[Residue],
        all_residues: list[Residue],
    ) -> list[_Dot]:
        """Generate valid solvent-accessible surface dots for *interface_residues*.

        Dots are placed on the extended Van der Waals sphere (r + probe) of
        each interface atom.  A dot is discarded when it lies inside any other
        atom's extended sphere (Lee-Richards occlusion criterion):
            dist(dot, atom_B) < r_B + probe_radius

        Occlusion is tested against *all_residues* (the full monomer chain).
        The outward normal at each dot is the unit vector from atom centre to dot.

        Parameters
        ----------
        interface_residues:
            Subset of residues whose surface we want (ΔSASA > 0).
        all_residues:
            All residues in the same chain (used for occlusion only).

        Returns
        -------
        list of (coord, normal) tuples, each a length-3 ndarray.
        """
        sphere_pts = self._fibonacci_sphere(self.n_sphere_dots)

        all_atoms = [
            a for r in all_residues for a in r.get_atoms()
            if self._atom_element(a) not in ("H", "D")
        ]
        if not all_atoms:
            return []

        all_coords = np.array([a.get_coord() for a in all_atoms])          # (M, 3)
        all_radii  = np.array(
            [self._atom_vdw(a) + self._PROBE_RADIUS for a in all_atoms]
        )                                                                    # (M,)

        dots: list[_Dot] = []
        for residue in interface_residues:
            for atom in residue.get_atoms():
                if self._atom_element(atom) in ("H", "D"):
                    continue
                center = atom.get_coord()
                r      = self._atom_vdw(atom) + self._PROBE_RADIUS

                dot_coords = center + r * sphere_pts                # (N, 3)

                # Vectorised occlusion test: dists[i, j] = dist(dot_i, atom_j)
                diffs  = dot_coords[:, None, :] - all_coords[None, :, :]  # (N, M, 3)
                dists  = np.linalg.norm(diffs, axis=2)                    # (N, M)
                valid  = np.all(dists >= all_radii[None, :], axis=1)      # (N,)

                for idx in np.where(valid)[0]:
                    dots.append((dot_coords[idx], sphere_pts[idx]))

        return dots

    def _sc_one_direction(
        self,
        dots_src: list[_Dot],
        dots_tgt: list[_Dot],
    ) -> Optional[float]:
        """Compute SC from *dots_src* toward *dots_tgt* (raw, before normalisation).

        For each source dot the nearest target dot is found with a cKDTree.
        Pairs within ``sc_dmax`` Å contribute ``n_src · (−n_tgt)`` to the
        accumulated list.  The *median* of this list is returned.

        Returns ``None`` when fewer than ``sc_min_pairs`` valid pairs are found.
        """
        if not dots_src or not dots_tgt:
            return None

        src_coords  = np.array([d[0] for d in dots_src])
        src_normals = np.array([d[1] for d in dots_src])
        tgt_coords  = np.array([d[0] for d in dots_tgt])
        tgt_normals = np.array([d[1] for d in dots_tgt])

        tree = cKDTree(tgt_coords)
        dists, indices = tree.query(src_coords, k=1, workers=-1)

        mask = dists <= self.sc_dmax
        if int(mask.sum()) < self.sc_min_pairs:
            return None

        # n_src · (−n_tgt): +1 for anti-parallel normals (complementary surfaces)
        dp = np.einsum(
            "ij,ij->i",
            src_normals[mask],
            -tgt_normals[indices[mask]],
        )
        return float(np.median(dp))

    @staticmethod
    def _normalise(sc_raw: float) -> float:
        """Map raw SC ∈ [−1, 1] to a normalised score ∈ [0, 1].

        Formula::

            sc_normalised = clip((sc_raw + 1) / 2, 0, 1)

        Interpretation:
            0.0  — perfectly anti-complementary (concave meets concave)
            0.5  — random / uncorrelated surface normals
            1.0  — perfect shape complementarity
        """
        return float(np.clip((sc_raw + 1.0) / 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        interface_residues_a: list[Residue],
        all_residues_a: list[Residue],
        interface_residues_b: list[Residue],
        all_residues_b: list[Residue],
    ) -> Optional[float]:
        """Compute the normalised SC score in [0, 1].

        Parameters
        ----------
        interface_residues_a:
            Interface residues of chain A (ΔSASA > 0) — from SASACalculator.
        all_residues_a:
            All protein residues of chain A (for occlusion checks).
        interface_residues_b:
            Interface residues of chain B.
        all_residues_b:
            All protein residues of chain B.

        Returns
        -------
        float in [0, 1], or ``None`` when too few dot pairs are found.
        """
        if not interface_residues_a or not interface_residues_b:
            return None

        dots_a = self._surface_dots(interface_residues_a, all_residues_a)
        dots_b = self._surface_dots(interface_residues_b, all_residues_b)

        sc_ab = self._sc_one_direction(dots_a, dots_b)
        sc_ba = self._sc_one_direction(dots_b, dots_a)

        valid = [s for s in (sc_ab, sc_ba) if s is not None]
        if not valid:
            return None

        sc_raw = sum(valid) / len(valid)
        return self._normalise(sc_raw)
