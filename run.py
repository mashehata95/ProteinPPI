"""
run.py — ProteinPPI main entry point
=====================================
Orchestrates the five analysis modules and writes output CSV files.

Output files
------------
interactions_<A><B>.csv
    One row per (residue pair × interaction type) between chains A and B.
    Only rows for detected interactions are written.

    Columns:
        chain_a, residue_a, dsasa_residue_a, pka_a,
        chain_b, residue_b, dsasa_residue_b, pka_b,
        interaction_type, atoms_a, atoms_b, distance, mediator

    ``dsasa_residue_*``  — ΔSASA (Å²) of the residue upon complex formation
    ``pka_a / pka_b``    — predicted pKa of the residue (ionisable residues only;
                           empty for non-ionisable residues or without --affinity)
    ``interaction_type`` — one of: HBond_donor, HBond_acceptor, Hydrophobic,
                           SaltBridge, PiStacking_Par, PiStacking_T, CationPi,
                           Disulfide, WaterBridge, VanDerWaal
    ``atoms_a``          — semicolon-separated atoms from chain A
    ``atoms_b``          — semicolon-separated atoms from chain B
    ``distance``         — minimum contact distance (Å)
    ``mediator``         — water label(s) for WaterBridge; empty otherwise

interface_metrics.csv
    One row per chain pair with SASA, BSA, normalised SC, and (optionally)
    the PRODIGY binding free energy and dissociation constant.

    Columns:
        chain_a, chain_b,
        sasa_chain_a_A2, sasa_chain_b_A2, sasa_complex_A2,
        bsa_A2, sc_normalised,
        delta_g_kcal_mol, kd_M   ← only present when --affinity is used

pka_<A><B>.csv   (only when --affinity is used)
    One row per ionisable residue in the pair with its predicted pKa.

    Columns:
        chain, residue, resnum, pka, model_pka

Usage
-----
Command-line::

    python run.py 1ABC.pdb -o results/

    # Intramolecular interactions for chain A:
    python run.py 1ABC.pdb -o results/ --intra-chain A

    # Binding affinity + pKa at pH 7.4 (default):
    python run.py 1ABC.pdb -o results/ --affinity

    # Affinity at a different pH and temperature:
    python run.py 1ABC.pdb -o results/ --affinity --pH 6.0 --temperature 37.0

Python API::

    from run import PPInteractions
    ppi = PPInteractions("1ABC.pdb")

    # Intermolecular only
    ppi.run(output_dir="results/")

    # With binding affinity + pKa at pH 7.4
    ppi.run(output_dir="results/", compute_affinity=True, pH=7.4)

    # Inspect in memory
    records = ppi.analyse_intra("A", min_seq_sep=4)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Optional

from proteinppi import (
    InteractionFinder,
    ProdigyCalculator,
    SAPCalculator,
    SASACalculator,
    ShapeComplementarity,
    StructureLoader,
)
from proteinppi.interactions import INTERACTION_TYPES


# ---------------------------------------------------------------------------
# CSV column definitions (single source of truth)
# ---------------------------------------------------------------------------

_INTERACTION_FIELDNAMES: list[str] = [
    "chain_a", "residue_a", "dsasa_residue_a", "pka_a",
    "chain_b", "residue_b", "dsasa_residue_b", "pka_b",
    "interaction_type", "atoms_a", "atoms_b", "distance", "mediator",
]

_METRICS_FIELDNAMES_BASE: list[str] = [
    "chain_a", "chain_b",
    "sasa_chain_a_A2", "sasa_chain_b_A2", "sasa_complex_A2",
    "bsa_A2", "sc_normalised",
]

_METRICS_FIELDNAMES_AFFINITY: list[str] = [
    *_METRICS_FIELDNAMES_BASE,
    "delta_g_kcal_mol",
    "kd_M",
]

_PKA_FIELDNAMES: list[str] = ProdigyCalculator.PKA_FIELDNAMES
_SAP_FIELDNAMES: list[str] = SAPCalculator.SAP_FIELDNAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resnum_from_label(residue_label: str) -> Optional[int]:
    """Extract the integer residue number from a label like 'ASP6' or 'LYS123A'."""
    m = re.search(r'\d+', residue_label)
    return int(m.group()) if m else None


def _annotate_pka(
    rows: list[dict],
    pka_dict: dict[tuple[str, int], dict],
) -> None:
    """Fill ``pka_a`` / ``pka_b`` fields in-place from *pka_dict*."""
    for row in rows:
        resnum_a = _resnum_from_label(row.get("residue_a", ""))
        resnum_b = _resnum_from_label(row.get("residue_b", ""))
        row["pka_a"] = (
            pka_dict.get((row["chain_a"], resnum_a), {}).get("pka", "")
            if resnum_a is not None else ""
        )
        row["pka_b"] = (
            pka_dict.get((row["chain_b"], resnum_b), {}).get("pka", "")
            if resnum_b is not None else ""
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PPInteractions:
    """Coordinate all analysis modules for every protein chain pair.

    Parameters
    ----------
    structure_file:
        Path to a ``.pdb``, ``.ent``, ``.cif``, or ``.mmcif`` file.
    n_sphere_dots:
        Dots per atom for SC surface generation (default 100).
    sc_dmax:
        Max dot-to-dot pairing distance (Å) for SC (default 4.0).
    """

    def __init__(
        self,
        structure_file: str,
        n_sphere_dots: int = 100,
        sc_dmax: float = 4.0,
    ) -> None:
        self.loader   = StructureLoader(structure_file)
        self.finder   = InteractionFinder(self.loader)
        self.sasa     = SASACalculator(self.loader)
        self.sc_calc  = ShapeComplementarity(
            self.loader,
            n_sphere_dots=n_sphere_dots,
            sc_dmax=sc_dmax,
        )
        self.affinity_calc: Optional[ProdigyCalculator] = None
        self.sap_calc = SAPCalculator(self.loader)

    # ------------------------------------------------------------------
    # Per-pair structural analysis
    # ------------------------------------------------------------------

    def analyse_pair(
        self,
        chain_a: str,
        chain_b: str,
        ca_prefilter: float = 12.0,
    ) -> tuple[list[dict], dict]:
        """Run the structural analysis pipeline for one chain pair.

        Returns
        -------
        interactions : list[dict]
            Records for the interaction CSV (``pka_a``/``pka_b`` are empty
            strings — filled later if ``--affinity`` is active).
        metrics : dict
            Record for the interface_metrics CSV.
        """
        print(f"\n  [{chain_a}–{chain_b}] Computing SASA / BSA …")
        sasa_result = self.sasa.compute_for_pair(chain_a, chain_b)

        print(f"  [{chain_a}–{chain_b}] Detecting interactions …")
        interactions = self.finder.find_all(
            chain_a, chain_b,
            dsasa_a=sasa_result["dsasa_a"],
            dsasa_b=sasa_result["dsasa_b"],
            ca_prefilter=ca_prefilter,
        )
        # Initialise pKa fields; filled later by _annotate_pka
        for row in interactions:
            row.setdefault("pka_a", "")
            row.setdefault("pka_b", "")

        print(f"  [{chain_a}–{chain_b}] Computing Shape Complementarity …")
        all_res_a = self.loader.protein_residues(chain_a)
        all_res_b = self.loader.protein_residues(chain_b)
        sc = self.sc_calc.compute(
            sasa_result["interface_residues_a"], all_res_a,
            sasa_result["interface_residues_b"], all_res_b,
        )

        metrics = {
            "chain_a":         chain_a,
            "chain_b":         chain_b,
            "sasa_chain_a_A2": sasa_result["sasa_chain_a"],
            "sasa_chain_b_A2": sasa_result["sasa_chain_b"],
            "sasa_complex_A2": sasa_result["sasa_complex"],
            "bsa_A2":          sasa_result["bsa"],
            "sc_normalised":   round(sc, 4) if sc is not None else "",
        }

        print(
            f"  [{chain_a}–{chain_b}]  "
            f"{len(interactions):>5} interactions | "
            f"BSA = {sasa_result['bsa']:.1f} Å² | "
            f"SC = {metrics['sc_normalised'] if metrics['sc_normalised'] != '' else 'N/A'}"
        )
        return interactions, metrics

    # ------------------------------------------------------------------
    # Intramolecular analysis
    # ------------------------------------------------------------------

    def analyse_intra(
        self,
        chain_id:     str,
        min_seq_sep:  int   = 4,
        ca_prefilter: float = 12.0,
    ) -> list[dict]:
        """Detect intramolecular interactions within a single chain.

        Parameters
        ----------
        chain_id:
            Chain identifier to analyse (must exist in the loaded structure).
        min_seq_sep:
            Minimum residue-index separation to consider a pair (default 4).
        ca_prefilter:
            Cα–Cα distance (Å) above which a pair is skipped.

        Returns
        -------
        list[dict]
            Interaction records (same schema as intermolecular output).
        """
        if chain_id not in self.loader.chain_ids:
            raise ValueError(
                f"Chain '{chain_id}' not found in structure. "
                f"Available chains: {self.loader.chain_ids}"
            )
        print(
            f"\n  [intra-{chain_id}] Detecting intramolecular interactions "
            f"(min_seq_sep={min_seq_sep}) …"
        )
        interactions = self.finder.find_intra(
            chain_id,
            min_seq_sep=min_seq_sep,
            ca_prefilter=ca_prefilter,
        )
        for row in interactions:
            row.setdefault("pka_a", "")
            row.setdefault("pka_b", "")
        print(f"  [intra-{chain_id}]  {len(interactions):>5} intramolecular interactions")
        return interactions

    # ------------------------------------------------------------------
    # Affinity + pKa
    # ------------------------------------------------------------------

    def analyse_affinity(
        self,
        chain_a: str,
        chain_b: str,
    ) -> dict:
        """Run PROPKA3 → PDB2PQR → PRODIGY for one chain pair.

        Returns
        -------
        dict — see :meth:`~proteinppi.affinity.ProdigyCalculator.compute_affinity`.
        """
        if self.affinity_calc is None:
            raise RuntimeError(
                "Affinity calculator not initialised.  "
                "Call run(compute_affinity=True) or set self.affinity_calc directly."
            )
        print(f"\n  [{chain_a}–{chain_b}] Computing binding affinity …")
        return self.affinity_calc.compute_affinity(chain_a, chain_b)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        output_dir:        str       = "results",
        ca_prefilter:      float     = 12.0,
        intra_chains:      list[str] = (),
        intra_min_seq_sep: int       = 4,
        compute_affinity:  bool      = False,
        pH:                float     = 7.4,
        temperature:       float     = 25.0,
        compute_sap:       bool      = False,
        sap_radius:        float     = 5.0,
    ) -> None:
        """Analyse all chain pairs (and optionally single chains) and write CSVs.

        Parameters
        ----------
        output_dir:
            Directory where all output files are written (created if absent).
        ca_prefilter:
            Cα–Cα distance threshold (Å) forwarded to InteractionFinder.
        intra_chains:
            Chain IDs for which intramolecular interactions are extracted.
        intra_min_seq_sep:
            Minimum residue-index separation for intramolecular pairs.
        compute_affinity:
            When ``True``, run PROPKA3 + PDB2PQR + PRODIGY for each pair.
            Requires ``propka``, ``pdb2pqr``, and ``prodigy-prot`` on PATH.
        pH:
            pH for structure protonation via PDB2PQR (default 7.4).
        temperature:
            Temperature (°C) for PRODIGY Kd prediction (default 25.0).
        compute_sap:
            When ``True``, compute Spatial Aggregation Propensity (SAP),
            SASA, and ΔSASA for every residue across all chains.
        sap_radius:
            Cβ sphere radius (Å) for SAP neighbour search (default 5.0).

        Output files
        ------------
        ``interactions_<A><B>.csv``
            One file per pair — intermolecular contacts with ΔSASA and pKa.
        ``intra_<chain>.csv``
            One file per requested chain — intramolecular contacts.
        ``interface_metrics.csv``
            SASA, BSA, SC, and (if affinity) ΔG and Kd per pair.
        ``pka_<A><B>.csv``
            pKa per ionisable residue for each pair (only with --affinity).
        ``sap.csv``
            SAP score, SASA, and ΔSASA for every residue (only with --sap).
        """
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nOutput directory : {output_dir}")
        if intra_chains:
            print(
                f"Intra chains     : {list(intra_chains)}"
                f"  (min_seq_sep={intra_min_seq_sep})"
            )
        print("-" * 60)

        # ---- Initialise affinity calculator (optional) ---------------
        if compute_affinity:
            try:
                self.affinity_calc = ProdigyCalculator(
                    self.loader,
                    pH=pH,
                    temperature=temperature,
                )
                print(
                    f"Affinity method  : PRODIGY  "
                    f"(pH={pH:.2f}, T={temperature}°C)"
                )
                print(f"pKa method       : PROPKA3")
            except RuntimeError as exc:
                print(f"WARNING: {exc}\nAffinity computation will be skipped.")
                compute_affinity = False

        if compute_sap:
            print(f"SAP method       : Shire et al. 2009  (radius={sap_radius} Å)")

        metrics_rows: list[dict] = []
        metrics_fieldnames = (
            _METRICS_FIELDNAMES_AFFINITY if compute_affinity
            else _METRICS_FIELDNAMES_BASE
        )

        for chain_a, chain_b in self.loader.chain_pairs:

            # ---- structural analysis --------------------------------
            interactions, metrics = self.analyse_pair(
                chain_a, chain_b, ca_prefilter=ca_prefilter,
            )

            # ---- affinity + pKa (optional) --------------------------
            if compute_affinity:
                try:
                    aff = self.analyse_affinity(chain_a, chain_b)

                    # Annotate interaction rows with pKa
                    _annotate_pka(interactions, aff["pka_dict"])

                    metrics["delta_g_kcal_mol"] = (
                        round(aff["delta_g_kcal_mol"], 3)
                        if aff["delta_g_kcal_mol"] is not None else ""
                    )
                    metrics["kd_M"] = (
                        f"{aff['kd_M']:.3e}"
                        if aff["kd_M"] is not None else ""
                    )

                    # pKa CSV
                    pka_path = os.path.join(
                        output_dir, f"pka_{chain_a}{chain_b}.csv"
                    )
                    with open(pka_path, "w", newline="") as fh:
                        writer = csv.DictWriter(fh, fieldnames=_PKA_FIELDNAMES)
                        writer.writeheader()
                        writer.writerows(aff["pka_records"])
                    print(f"  Saved → {pka_path}")

                except RuntimeError as exc:
                    print(
                        f"  WARNING: affinity computation for "
                        f"{chain_a}-{chain_b} failed: {exc}"
                    )
                    metrics["delta_g_kcal_mol"] = ""
                    metrics["kd_M"] = ""

            # ---- interaction CSV ------------------------------------
            out_path = os.path.join(
                output_dir, f"interactions_{chain_a}{chain_b}.csv"
            )
            with open(out_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_INTERACTION_FIELDNAMES)
                writer.writeheader()
                writer.writerows(interactions)
            print(f"  Saved → {out_path}")

            metrics_rows.append(metrics)

        # ---- intramolecular interactions ----------------------------
        for chain_id in intra_chains:
            try:
                intra_records = self.analyse_intra(
                    chain_id,
                    min_seq_sep=intra_min_seq_sep,
                    ca_prefilter=ca_prefilter,
                )
            except ValueError as exc:
                print(f"  WARNING: {exc}")
                continue

            # pKa annotation for intra chains (if affinity is active)
            if compute_affinity and self.affinity_calc is not None:
                try:
                    pka_dict, _ = self.affinity_calc.compute_pka([chain_id])
                    _annotate_pka(intra_records, pka_dict)
                except RuntimeError as exc:
                    print(f"  WARNING: pKa for intra-{chain_id} failed: {exc}")

            intra_path = os.path.join(output_dir, f"intra_{chain_id}.csv")
            with open(intra_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_INTERACTION_FIELDNAMES)
                writer.writeheader()
                writer.writerows(intra_records)
            print(f"  Saved → {intra_path}")

        # ---- interface metrics CSV ----------------------------------
        metrics_path = os.path.join(output_dir, "interface_metrics.csv")
        with open(metrics_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=metrics_fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"  Saved → {metrics_path}")

        # ---- SAP CSV (optional) -------------------------------------
        if compute_sap:
            print(f"\n  Computing SAP scores (radius={sap_radius} Å) …")
            try:
                sap_records = self.sap_calc.compute(radius=sap_radius)
                sap_path = os.path.join(output_dir, "sap.csv")
                with open(sap_path, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=_SAP_FIELDNAMES)
                    writer.writeheader()
                    writer.writerows(sap_records)
                print(f"  {len(sap_records)} residues scored → {sap_path}")
            except Exception as exc:
                print(f"  WARNING: SAP computation failed: {exc}")

        print("-" * 60)
        print(f"All output written to: {output_dir}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python run.py",
        description=(
            "ProteinPPI — detect protein-protein interactions and compute "
            "interface metrics (BSA, SC, optionally binding affinity via PRODIGY "
            "and pKa via PROPKA3) for every chain pair."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "structure",
        help="Input structure file (.pdb, .ent, .cif, .mmcif).",
    )
    p.add_argument(
        "-o", "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for output CSV files (created if absent). "
            "Defaults to '<input_stem>_ppi/' next to the input file."
        ),
    )
    p.add_argument(
        "--ca-prefilter",
        type=float,
        default=12.0,
        metavar="Å",
        help="Cα–Cα distance threshold (Å) for the residue-pair pre-filter.",
    )
    p.add_argument(
        "--n-sphere-dots",
        type=int,
        default=100,
        metavar="N",
        help="Fibonacci sphere dots per atom for SC surface generation.",
    )
    p.add_argument(
        "--sc-dmax",
        type=float,
        default=4.0,
        metavar="Å",
        help="Maximum dot-to-dot distance (Å) for SC pairing.",
    )

    # ---- intramolecular group ---------------------------------------
    ig = p.add_argument_group("Intramolecular interactions")
    ig.add_argument(
        "--intra-chain",
        nargs="+",
        metavar="CHAIN_ID",
        default=[],
        dest="intra_chains",
        help=(
            "Extract intramolecular interactions within the specified chain(s). "
            "One output file per chain: intra_<chain>.csv.  "
            "Example: --intra-chain A B"
        ),
    )
    ig.add_argument(
        "--intra-min-seq-sep",
        type=int,
        default=4,
        metavar="N",
        dest="intra_min_seq_sep",
        help=(
            "Minimum residue-index separation for intramolecular pairs.  "
            "Pairs closer than this are skipped (default 4)."
        ),
    )

    # ---- SAP group --------------------------------------------------
    sg = p.add_argument_group("Spatial Aggregation Propensity (SAP)")
    sg.add_argument(
        "--sap",
        action="store_true",
        default=False,
        dest="compute_sap",
        help=(
            "Compute SAP score, SASA, and ΔSASA for every residue across all "
            "chains.  Writes sap.csv to the output directory.  "
            "Uses the Shire et al. (2009) formula with the Black-Mould (1991) "
            "hydrophobicity scale."
        ),
    )
    sg.add_argument(
        "--sap-radius",
        type=float,
        default=5.0,
        metavar="Å",
        dest="sap_radius",
        help=(
            "Sphere radius (Å) centred on Cβ (Cα for Gly) for SAP neighbour "
            "search (default 5.0)."
        ),
    )

    # ---- affinity / pKa group ---------------------------------------
    ag = p.add_argument_group(
        "Binding affinity and pKa  "
        "(requires: pip install propka pdb2pqr prodigy-prot)"
    )
    ag.add_argument(
        "--affinity",
        action="store_true",
        default=False,
        help=(
            "Compute binding affinity (PRODIGY) and per-residue pKa (PROPKA3) "
            "for each chain pair.  Adds delta_g_kcal_mol and kd_M to "
            "interface_metrics.csv, pka_a/pka_b columns to the interaction "
            "CSV, and produces pka_<AB>.csv files."
        ),
    )
    ag.add_argument(
        "--pH",
        type=float,
        default=7.4,
        metavar="FLOAT",
        dest="pH",
        help=(
            "pH used by PDB2PQR to set protonation states before PRODIGY "
            "(default 7.4).  Affects which residues are classified as charged."
        ),
    )
    ag.add_argument(
        "--temperature",
        type=float,
        default=25.0,
        metavar="FLOAT",
        dest="temperature",
        help="Temperature (°C) passed to PRODIGY for Kd calculation (default 25.0).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        stem       = os.path.splitext(os.path.basename(args.structure))[0]
        output_dir = os.path.abspath(os.path.join(
            os.path.dirname(args.structure), f"{stem}_ppi"
        ))

    ppi = PPInteractions(
        args.structure,
        n_sphere_dots=args.n_sphere_dots,
        sc_dmax=args.sc_dmax,
    )
    ppi.run(
        output_dir=output_dir,
        ca_prefilter=args.ca_prefilter,
        intra_chains=args.intra_chains,
        intra_min_seq_sep=args.intra_min_seq_sep,
        compute_affinity=args.affinity,
        pH=args.pH,
        temperature=args.temperature,
        compute_sap=args.compute_sap,
        sap_radius=args.sap_radius,
    )


if __name__ == "__main__":
    main()
