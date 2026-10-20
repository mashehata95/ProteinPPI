"""
interactions.py — InteractionFinder
=====================================
Detects ten types of protein-protein interactions between residue pairs from
two chains and returns **one record per (residue pair × interaction type)**.

Interaction types
-----------------
HBond_donor     res_a donates an H-bond to res_b (donor N/O on A, acceptor O/N on B) ≤ 3.5 Å
HBond_acceptor  res_a accepts an H-bond from res_b (acceptor on A, donor on B) ≤ 3.5 Å
Hydrophobic     Closest non-polar atom pair ≤ 4.0 Å
SaltBridge      Oppositely charged side-chain atoms ≤ 5.5 Å
PiStacking_Par  Parallel aromatic rings: centroid dist ≤ 5.5 Å, angle ≤ 30°
PiStacking_T    T-shaped aromatic rings: centroid dist ≤ 5.5 Å, angle 60–90°
CationPi        Cation to ring centroid ≤ 6.0 Å, within 30° of ring normal
Disulfide       Cys SG–SG ≤ 2.5 Å
WaterBridge     Water O within 3.5 Å of H-bond-capable atoms on both residues
VanDerWaal      Any heavy-atom pair closest contact ≤ 4.0 Å

Output columns (one row per residue-pair × interaction-type)
------------------------------------------------------------
chain_a, residue_a, dsasa_residue_a,
chain_b, residue_b, dsasa_residue_b,
interaction_type, atoms_a, atoms_b, distance, mediator

``atoms_a`` / ``atoms_b``
    Semicolon-separated list of **unique** atom names from the respective
    residue that participate in this interaction type for this pair.
    e.g. ``"NZ"`` / ``"OE1;OE2"``  or  ``"ring"`` / ``"ring"``

``distance``
    Minimum heavy-atom contact distance (Å) among all hits of this type for
    this residue pair, rounded to 3 decimal places.
    For WaterBridge: minimum of (res_a→water, res_b→water) distances.

``mediator``
    Water residue label for WaterBridge rows (e.g. ``"HOH401"``); empty for
    all other interaction types.  When multiple bridging waters are found,
    labels are semicolon-separated.

Invariant
---------
``atoms_a`` always contains atoms belonging to chain A (``res_a``).
``atoms_b`` always contains atoms belonging to chain B (``res_b``).
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue

from .structure import StructureLoader

# ---------------------------------------------------------------------------
# Ordered list of all interaction types (used as valid interaction_type values)
# ---------------------------------------------------------------------------
INTERACTION_TYPES: list[str] = [
    "HBond_donor", "HBond_acceptor", "Hydrophobic", "SaltBridge",
    "PiStacking_Par", "PiStacking_T", "CationPi", "Disulfide",
    "WaterBridge", "VanDerWaal",
]

# ---------------------------------------------------------------------------
# Internal hit type
# ---------------------------------------------------------------------------
# Each finder returns a list of _Hit dicts.
# Keys: itype (str), atom_a (str), atom_b (str), distance (float), mediator (str)
_Hit = dict  # typed alias for readability


class InteractionFinder:
    """Detect all interaction types between two protein chains.

    Parameters
    ----------
    loader:
        A fully initialised :class:`~proteinppi.structure.StructureLoader`.
    """

    # ------------------------------------------------------------------
    # Interaction geometry constants
    # ------------------------------------------------------------------

    _SC_DONORS: dict[str, list[str]] = {
        "ARG": ["NE", "NH1", "NH2"], "ASN": ["ND2"], "GLN": ["NE2"],
        "HIS": ["ND1", "NE2"], "LYS": ["NZ"], "SER": ["OG"],
        "THR": ["OG1"], "TRP": ["NE1"], "TYR": ["OH"], "CYS": ["SG"],
    }

    _SC_ACCEPTORS: dict[str, list[str]] = {
        "ASN": ["OD1"], "ASP": ["OD1", "OD2"], "GLN": ["OE1"],
        "GLU": ["OE1", "OE2"], "HIS": ["ND1", "NE2"], "MET": ["SD"],
        "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"],
    }

    _AROMATIC_RINGS: dict[str, list[list[str]]] = {
        "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
        "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
        "TRP": [
            ["CG", "CD1", "NE1", "CE2", "CD2"],
            ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        ],
        "HIS": [["CG", "ND1", "CE1", "NE2", "CD2"]],
    }

    _POS_CHARGED: dict[str, list[str]] = {
        "ARG": ["NH1", "NH2", "NE"], "LYS": ["NZ"], "HIS": ["ND1", "NE2"],
    }

    _NEG_CHARGED: dict[str, list[str]] = {
        "ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"],
    }

    _HYDROPHOBIC: dict[str, list[str]] = {
        "ALA": ["CB"], "VAL": ["CB", "CG1", "CG2"],
        "LEU": ["CB", "CG", "CD1", "CD2"], "ILE": ["CB", "CG1", "CG2", "CD1"],
        "MET": ["CG", "SD", "CE"],
        "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "PRO": ["CB", "CG", "CD"],
        "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2"],
        "CYS": ["CB", "SG"],
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, loader: StructureLoader) -> None:
        self.loader = loader

    # ------------------------------------------------------------------
    # Atom-level helpers
    # ------------------------------------------------------------------

    def _get_atoms(self, residue: Residue, names) -> list[Atom]:
        atoms = []
        for name in names:
            try:
                atoms.append(residue[name])
            except KeyError:
                pass
        return atoms

    def _donor_atoms(self, residue: Residue) -> list[Atom]:
        """Return H-bond donor heavy atoms.  Backbone N excluded for PRO."""
        names: list[str] = [] if residue.resname == "PRO" else ["N"]
        names += self._SC_DONORS.get(residue.resname, [])
        return self._get_atoms(residue, names)

    def _acceptor_atoms(self, residue: Residue) -> list[Atom]:
        """Return H-bond acceptor heavy atoms (backbone O always included)."""
        names: list[str] = ["O"] + self._SC_ACCEPTORS.get(residue.resname, [])
        return self._get_atoms(residue, dict.fromkeys(names))

    def _ring_geometry(
        self, residue: Residue, ring_atom_names: list[str]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (centroid, unit-normal) for an aromatic ring via SVD."""
        coords = []
        for name in ring_atom_names:
            try:
                coords.append(residue[name].get_coord())
            except KeyError:
                pass
        if len(coords) < 3:
            return None, None
        arr = np.array(coords)
        centroid = arr.mean(axis=0)
        _, _, vh = np.linalg.svd(arr - centroid)
        normal = vh[-1] / np.linalg.norm(vh[-1])
        return centroid, normal

    @staticmethod
    def _is_heavy(atom: Atom) -> bool:
        """Return True when *atom* is a non-hydrogen heavy atom."""
        elem = (atom.element or "").strip().upper()
        if elem:
            return elem not in ("H", "D")
        # Fall back to atom name when element tag is absent
        name = atom.get_name().strip().lstrip("0123456789")
        return bool(name) and name[0].upper() != "H"

    @staticmethod
    def _make_hit(
        itype: str, atom_a: str, atom_b: str,
        distance: float, mediator: str = "",
    ) -> _Hit:
        return {
            "itype":    itype,
            "atom_a":   atom_a,
            "atom_b":   atom_b,
            "distance": distance,
            "mediator": mediator,
        }

    # ------------------------------------------------------------------
    # Individual interaction finders
    # All finders guarantee: atom_a ∈ res_a, atom_b ∈ res_b
    # ------------------------------------------------------------------

    def _find_hbond_donor(
        self, res_a: Residue, res_b: Residue, cutoff: float = 3.5
    ) -> list[_Hit]:
        """H-bond where res_a is the donor.

        atom_a = donor atom on res_a (N, NZ, …)
        atom_b = acceptor atom on res_b (O, OD1, …)
        """
        hits: list[_Hit] = []
        for d in self._donor_atoms(res_a):
            for a in self._acceptor_atoms(res_b):
                dist = StructureLoader.dist(d.get_coord(), a.get_coord())
                if dist <= cutoff:
                    hits.append(self._make_hit(
                        "HBond_donor", d.get_name(), a.get_name(), dist
                    ))
        return hits

    def _find_hbond_acceptor(
        self, res_a: Residue, res_b: Residue, cutoff: float = 3.5
    ) -> list[_Hit]:
        """H-bond where res_a is the acceptor (res_b donates).

        atom_a = acceptor atom on res_a (O, OD1, …)
        atom_b = donor atom on res_b (N, NZ, …)
        """
        hits: list[_Hit] = []
        for d in self._donor_atoms(res_b):
            for a in self._acceptor_atoms(res_a):
                dist = StructureLoader.dist(d.get_coord(), a.get_coord())
                if dist <= cutoff:
                    hits.append(self._make_hit(
                        "HBond_acceptor", a.get_name(), d.get_name(), dist
                    ))
        return hits

    def _find_hydrophobic(
        self, res_a: Residue, res_b: Residue, cutoff: float = 4.0
    ) -> list[_Hit]:
        """Hydrophobic: closest non-polar atom pair ≤ *cutoff* Å (one hit per pair)."""
        atoms_a = self._get_atoms(res_a, self._HYDROPHOBIC.get(res_a.resname, []))
        atoms_b = self._get_atoms(res_b, self._HYDROPHOBIC.get(res_b.resname, []))
        best_dist, best_pair = float("inf"), None
        for at_a in atoms_a:
            for at_b in atoms_b:
                dist = StructureLoader.dist(at_a.get_coord(), at_b.get_coord())
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (at_a.get_name(), at_b.get_name())
        if best_pair and best_dist <= cutoff:
            return [self._make_hit("Hydrophobic", best_pair[0], best_pair[1], best_dist)]
        return []

    def _find_salt_bridges(
        self, res_a: Residue, res_b: Residue, cutoff: float = 5.5
    ) -> list[_Hit]:
        """Salt bridges: oppositely charged atoms ≤ *cutoff* Å.

        Both charge orderings:
        • positive on res_a / negative on res_b → atom_a = pos_a, atom_b = neg_b
        • negative on res_a / positive on res_b → atom_a = neg_a, atom_b = pos_b
        """
        hits: list[_Hit] = []
        pos_a = self._get_atoms(res_a, self._POS_CHARGED.get(res_a.resname, []))
        neg_a = self._get_atoms(res_a, self._NEG_CHARGED.get(res_a.resname, []))
        pos_b = self._get_atoms(res_b, self._POS_CHARGED.get(res_b.resname, []))
        neg_b = self._get_atoms(res_b, self._NEG_CHARGED.get(res_b.resname, []))
        for a_atoms, b_atoms in [(pos_a, neg_b), (neg_a, pos_b)]:
            for at_a in a_atoms:
                for at_b in b_atoms:
                    dist = StructureLoader.dist(at_a.get_coord(), at_b.get_coord())
                    if dist <= cutoff:
                        hits.append(self._make_hit(
                            "SaltBridge", at_a.get_name(), at_b.get_name(), dist
                        ))
        return hits

    def _find_pi_stacking(
        self, res_a: Residue, res_b: Residue,
        dist_cutoff: float = 5.5, par_angle: float = 30.0,
    ) -> list[_Hit]:
        """π–π stacking: centroid distance ≤ *dist_cutoff* Å, angle criterion.

        Returns PiStacking_Par or PiStacking_T hits; atom_a = atom_b = "ring".
        """
        hits: list[_Hit] = []
        for ring_a in self._AROMATIC_RINGS.get(res_a.resname, []):
            c_a, n_a = self._ring_geometry(res_a, ring_a)
            if c_a is None:
                continue
            for ring_b in self._AROMATIC_RINGS.get(res_b.resname, []):
                c_b, n_b = self._ring_geometry(res_b, ring_b)
                if c_b is None:
                    continue
                dist  = StructureLoader.dist(c_a, c_b)
                if dist > dist_cutoff:
                    continue
                cos   = float(abs(np.dot(n_a, n_b)))
                angle = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))
                if angle <= par_angle:
                    hits.append(self._make_hit("PiStacking_Par", "ring", "ring", dist))
                elif angle >= (90.0 - par_angle):
                    hits.append(self._make_hit("PiStacking_T", "ring", "ring", dist))
        return hits

    def _find_cation_pi(
        self, res_a: Residue, res_b: Residue,
        dist_cutoff: float = 6.0, angle_cutoff: float = 30.0,
    ) -> list[_Hit]:
        """Cation–π: cation to ring centroid ≤ *dist_cutoff* Å, angle ≤ *angle_cutoff*°.

        Both orderings:
        • cation on res_a, ring on res_b → atom_a = cation atom, atom_b = "ring"
        • cation on res_b, ring on res_a → atom_a = "ring",       atom_b = cation atom
        """
        hits: list[_Hit] = []

        # Case 1: cation from res_a, ring from res_b
        for cat in self._get_atoms(res_a, self._POS_CHARGED.get(res_a.resname, [])):
            for ring_b in self._AROMATIC_RINGS.get(res_b.resname, []):
                centroid, normal = self._ring_geometry(res_b, ring_b)
                if centroid is None:
                    continue
                vec  = cat.get_coord() - centroid
                dist = float(np.linalg.norm(vec))
                if dist > dist_cutoff:
                    continue
                cos   = float(abs(np.dot(vec / dist, normal)))
                angle = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))
                if angle <= angle_cutoff:
                    hits.append(self._make_hit("CationPi", cat.get_name(), "ring", dist))

        # Case 2: cation from res_b, ring from res_a
        for cat in self._get_atoms(res_b, self._POS_CHARGED.get(res_b.resname, [])):
            for ring_a in self._AROMATIC_RINGS.get(res_a.resname, []):
                centroid, normal = self._ring_geometry(res_a, ring_a)
                if centroid is None:
                    continue
                vec  = cat.get_coord() - centroid
                dist = float(np.linalg.norm(vec))
                if dist > dist_cutoff:
                    continue
                cos   = float(abs(np.dot(vec / dist, normal)))
                angle = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))
                if angle <= angle_cutoff:
                    hits.append(self._make_hit("CationPi", "ring", cat.get_name(), dist))

        return hits

    def _find_disulfide(
        self, res_a: Residue, res_b: Residue, cutoff: float = 2.5
    ) -> list[_Hit]:
        """Disulfide bond: Cys SG–SG ≤ *cutoff* Å."""
        if res_a.resname != "CYS" or res_b.resname != "CYS":
            return []
        try:
            dist = StructureLoader.dist(
                res_a["SG"].get_coord(), res_b["SG"].get_coord()
            )
            if dist <= cutoff:
                return [self._make_hit("Disulfide", "SG", "SG", dist)]
        except KeyError:
            pass
        return []

    def _find_water_bridges(
        self, res_a: Residue, res_b: Residue, cutoff: float = 3.5
    ) -> list[_Hit]:
        """Water bridge: water O within *cutoff* Å of H-bond atoms on **both** residues.

        For each bridging water:
        • atom_a = closest H-bond-capable atom on res_a
        • atom_b = closest H-bond-capable atom on res_b
        • distance = min(dist_a, dist_b)  — the tighter constraint
        • mediator = water residue label (e.g. "HOH401")
        """
        hb_a = self._donor_atoms(res_a) + self._acceptor_atoms(res_a)
        hb_b = self._donor_atoms(res_b) + self._acceptor_atoms(res_b)
        if not hb_a or not hb_b:
            return []

        hits: list[_Hit] = []
        for water in self.loader.waters:
            try:
                wat_coord = water["O"].get_coord()
            except KeyError:
                continue

            contacts_a = [
                (StructureLoader.dist(a.get_coord(), wat_coord), a)
                for a in hb_a
                if StructureLoader.dist(a.get_coord(), wat_coord) <= cutoff
            ]
            if not contacts_a:
                continue
            contacts_b = [
                (StructureLoader.dist(a.get_coord(), wat_coord), a)
                for a in hb_b
                if StructureLoader.dist(a.get_coord(), wat_coord) <= cutoff
            ]
            if not contacts_b:
                continue

            dist_a, atom_a = min(contacts_a, key=lambda x: x[0])
            dist_b, atom_b = min(contacts_b, key=lambda x: x[0])
            mediator = f"HOH{water.get_id()[1]}"
            hits.append(self._make_hit(
                "WaterBridge",
                atom_a.get_name(),
                atom_b.get_name(),
                min(dist_a, dist_b),
                mediator,
            ))
        return hits

    def _find_vdw(
        self, res_a: Residue, res_b: Residue, cutoff: float = 4.0
    ) -> list[_Hit]:
        """Van der Waals contact: closest heavy-atom pair ≤ *cutoff* Å.

        Reports the single shortest heavy-atom contact between the two
        residues regardless of atom type.  This captures generic close
        packing not classified by the more specific interaction types.
        """
        best_dist = float("inf")
        best_pair: Optional[tuple[str, str]] = None

        for at_a in res_a.get_atoms():
            if not self._is_heavy(at_a):
                continue
            for at_b in res_b.get_atoms():
                if not self._is_heavy(at_b):
                    continue
                dist = StructureLoader.dist(at_a.get_coord(), at_b.get_coord())
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (at_a.get_name(), at_b.get_name())

        if best_pair and best_dist <= cutoff:
            return [self._make_hit("VanDerWaal", best_pair[0], best_pair[1], best_dist)]
        return []

    # ------------------------------------------------------------------
    # Finder registry
    # ------------------------------------------------------------------

    _FINDERS: list[str] = [
        "_find_hbond_donor",
        "_find_hbond_acceptor",
        "_find_hydrophobic",
        "_find_salt_bridges",
        "_find_pi_stacking",
        "_find_cation_pi",
        "_find_disulfide",
        "_find_water_bridges",
        "_find_vdw",
    ]

    # ------------------------------------------------------------------
    # Shared record-building helper
    # ------------------------------------------------------------------

    def _build_records_for_pair(
        self,
        ra: Residue,
        rb: Residue,
        chain_a_id: str,
        label_a:    str,
        dsasa_a_val: str,
        chain_b_id: str,
        label_b:    str,
        dsasa_b_val: str,
    ) -> list[dict]:
        """Run every finder on *(ra, rb)* and return one record per itype found.

        This is the shared inner loop used by both :meth:`find_all` (intermolecular)
        and :meth:`find_intra` (intramolecular).  Callers are responsible for
        the residue-iteration strategy and Cα pre-filter.

        Parameters
        ----------
        ra, rb:
            The two residues to analyse.
        chain_a_id, label_a, dsasa_a_val:
            Chain ID, residue label, and ΔSASA string for residue A.
        chain_b_id, label_b, dsasa_b_val:
            Same for residue B.

        Returns
        -------
        list[dict] — zero or more records (one per detected interaction type).
        """
        type_hits: dict[str, list[_Hit]] = {}
        for finder_name in self._FINDERS:
            for hit in getattr(self, finder_name)(ra, rb):
                type_hits.setdefault(hit["itype"], []).append(hit)

        if not type_hits:
            return []

        records: list[dict] = []
        for itype in INTERACTION_TYPES:     # canonical order
            hits = type_hits.get(itype)
            if not hits:
                continue

            records.append({
                "chain_a":          chain_a_id,
                "residue_a":        label_a,
                "dsasa_residue_a":  dsasa_a_val,
                "chain_b":          chain_b_id,
                "residue_b":        label_b,
                "dsasa_residue_b":  dsasa_b_val,
                "interaction_type": itype,
                "atoms_a":          ";".join(dict.fromkeys(h["atom_a"] for h in hits)),
                "atoms_b":          ";".join(dict.fromkeys(h["atom_b"] for h in hits)),
                "distance":         round(min(h["distance"] for h in hits), 3),
                "mediator":         ";".join(dict.fromkeys(
                    h["mediator"] for h in hits if h["mediator"]
                )),
            })
        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_all(
        self,
        chain_a_id: str,
        chain_b_id: str,
        dsasa_a: Optional[dict[str, float]] = None,
        dsasa_b: Optional[dict[str, float]] = None,
        ca_prefilter: float = 12.0,
    ) -> list[dict]:
        """Return one record per (residue pair × interaction type) — **intermolecular**.

        Iterates over the Cartesian product of the two chains' residues.
        For each pair, every interaction type is tested; rows are only
        emitted for types that were actually detected.

        Parameters
        ----------
        chain_a_id, chain_b_id:
            Chain identifiers present in the loaded structure.
        dsasa_a:
            Per-residue ΔSASA for chain A (residue label → Å²).
        dsasa_b:
            Same for chain B.
        ca_prefilter:
            Cα–Cα distance (Å) above which a residue pair is skipped.

        Returns
        -------
        list[dict]
            One dict per (residue pair, interaction type) with keys matching
            :data:`INTERACTION_TYPES` plus identity / ΔSASA columns.
        """
        dsasa_a = dsasa_a or {}
        dsasa_b = dsasa_b or {}

        res_a_list = self.loader.protein_residues(chain_a_id)
        res_b_list = self.loader.protein_residues(chain_b_id)
        records: list[dict] = []

        for ra, rb in itertools.product(res_a_list, res_b_list):
            ca_a = self.loader.ca_coord(ra)
            ca_b = self.loader.ca_coord(rb)
            if (
                ca_a is not None and ca_b is not None
                and StructureLoader.dist(ca_a, ca_b) > ca_prefilter
            ):
                continue

            label_a = self.loader.res_label(ra)
            label_b = self.loader.res_label(rb)

            records.extend(self._build_records_for_pair(
                ra, rb,
                chain_a_id, label_a, str(dsasa_a.get(label_a, "")),
                chain_b_id, label_b, str(dsasa_b.get(label_b, "")),
            ))

        return records

    def find_intra(
        self,
        chain_id:    str,
        min_seq_sep: int   = 4,
        ca_prefilter: float = 12.0,
    ) -> list[dict]:
        """Return one record per (residue pair × interaction type) — **intramolecular**.

        Iterates over all unique residue pairs *within* a single chain using
        ``itertools.combinations``, so each pair is tested exactly once
        (ordered by sequence position, ra always before rb).

        Pairs whose sequence-index separation is less than *min_seq_sep* are
        skipped to exclude trivially bonded or angle-connected neighbours.
        The default of **4** captures:

        * α-helix backbone H-bonds (i → i+4)
        * β-sheet H-bonds (any separation ≥ 4)
        * Tertiary packing contacts (salt bridges, hydrophobic clusters, etc.)

        while excluding the directly bonded (i,i+1), angle-bonded (i,i+2),
        and dihedral-bonded (i,i+3) pairs that would generate spurious hits.

        Parameters
        ----------
        chain_id:
            Chain identifier to analyse (must exist in the loaded structure).
        min_seq_sep:
            Minimum residue-list index difference required to include a pair
            (default 4).
        ca_prefilter:
            Cα–Cα distance (Å) above which a pair is skipped (default 12.0 Å).

        Returns
        -------
        list[dict]
            One dict per (residue pair, interaction type), same column schema
            as :meth:`find_all`.  ``chain_a`` and ``chain_b`` are both set to
            *chain_id*.  ``dsasa_residue_a`` and ``dsasa_residue_b`` are
            empty strings (ΔSASA is not meaningful for intramolecular pairs).
        """
        residues = self.loader.protein_residues(chain_id)
        records:  list[dict] = []

        for i, j in itertools.combinations(range(len(residues)), 2):
            if j - i < min_seq_sep:
                continue

            ra, rb = residues[i], residues[j]

            ca_a = self.loader.ca_coord(ra)
            ca_b = self.loader.ca_coord(rb)
            if (
                ca_a is not None and ca_b is not None
                and StructureLoader.dist(ca_a, ca_b) > ca_prefilter
            ):
                continue

            label_a = self.loader.res_label(ra)
            label_b = self.loader.res_label(rb)

            records.extend(self._build_records_for_pair(
                ra, rb,
                chain_id, label_a, "",   # dsasa not applicable intramolecularly
                chain_id, label_b, "",
            ))

        return records
