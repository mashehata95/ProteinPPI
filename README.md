# ProteinPPI

**ProteinPPI** is a modular Python tool for extracting protein–protein interactions (PPIs) from static Experimental or Predicted structures stored in PDB or mmCIF format.  
It is inspired by [ProLIF](https://github.com/chemosim-lab/ProLIF) but focuses exclusively on protein–protein contacts and requires no RDKit dependency.

---

## Project Structure

```
ProteinPPI/
├── run.py                          ← main entry point (CLI + orchestrator)
├── requirements.txt
├── README.md
└── proteinppi/                     ← analysis modules
    ├── __init__.py
    ├── structure.py                ← StructureLoader        (file parsing, chain helpers)
    ├── sasa.py                     ← SASACalculator         (SASA, BSA, ΔSASA per residue)
    ├── interactions.py             ← InteractionFinder      (10 interaction types)
    └── shape_complementarity.py    ← ShapeComplementarity   (SC, normalised 0→1)
```

Each module has a single responsibility and can be imported and used independently.

---

## Detected Interaction Types

| Type | Geometric Criterion |
|---|---|
| `HBond_donor` | res_A donates H-bond to res_B: donor–acceptor heavy-atom distance ≤ 3.5 Å |
| `HBond_acceptor` | res_A accepts H-bond from res_B: donor–acceptor heavy-atom distance ≤ 3.5 Å |
| `Hydrophobic` | Closest non-polar atom pair ≤ 4.0 Å |
| `SaltBridge` | Oppositely charged side-chain atoms ≤ 5.5 Å |
| `PiStacking_Par` | Parallel aromatic rings: centroid distance ≤ 5.5 Å and inter-normal angle ≤ 30° |
| `PiStacking_T` | T-shaped aromatic rings: centroid distance ≤ 5.5 Å and inter-normal angle 60–90° |
| `CationPi` | Cation to ring centroid ≤ 6.0 Å, within 30° of ring normal |
| `Disulfide` | Cys SG–SG distance ≤ 2.5 Å |
| `WaterBridge` | Water O within 3.5 Å of H-bond-capable atoms on **both** residues simultaneously |
| `VanDerWaal` | Closest heavy-atom pair ≤ 4.0 Å (generic close contact) |

---

## Requirements

### Core (always required)

- Python ≥ 3.10
- [Biopython](https://biopython.org/) ≥ 1.79
- [NumPy](https://numpy.org/) ≥ 1.21
- [freesasa](https://freesasa.github.io/) ≥ 2.0 — SASA computation for BSA and ΔSASA
- [SciPy](https://scipy.org/) ≥ 1.7 — `cKDTree` for fast SC nearest-neighbour search

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
git clone https://github.com/<your-username>/ProteinPPI.git
cd ProteinPPI
pip install -r requirements.txt
```

No build steps are required — all modules are plain Python files.

---

## Usage

### Command-line

```bash
python run.py <structure_file> [options]
```

| Argument | Default | Description |
|---|---|---|
| `structure` | — | Input structure (`.pdb`, `.ent`, `.cif`, `.mmcif`) |
| `-o`, `--output-dir` | `<stem>_ppi/` | Directory for output CSVs |
| `--ca-prefilter` | `12.0` | Cα–Cα distance pre-filter (Å) |
| `--n-sphere-dots` | `100` | Fibonacci sphere dots per atom for SC |
| `--sc-dmax` | `4.0` | Max dot-to-dot pairing distance (Å) for SC |
| `--intra-chain` | — | Chain ID(s) for intramolecular interaction extraction (e.g. `A` or `A B C`) |

#### Examples

```bash
# Analyse a PDB file, write CSVs to <stem>_ppi/
python run.py 1ABC.pdb

# Specify output directory
python run.py 1ABC.pdb -o results/

# mmCIF input, higher SC resolution, custom output directory
python run.py 1ABC.cif -o results/ --n-sphere-dots 200 --sc-dmax 3.5

# Intramolecular interactions for chain A → intra_A.csv
python run.py 1ABC.pdb -o results/ --intra-chain A

# Intramolecular for multiple chains simultaneously
python run.py 1ABC.pdb -o results/ --intra-chain A B C

# Looser sequence-separation filter (captures γ-turns at i→i+3)
python run.py 1ABC.pdb -o results/ --intra-chain A --intra-min-seq-sep 3


```

### Python API

```python
from run import PPInteractions

ppi = PPInteractions("1ABC.pdb")

# Structural analysis only → interactions_AB.csv + interface_metrics.csv
ppi.run(output_dir="results/")

# Analyse a single chain pair and inspect results in memory
interactions, metrics = ppi.analyse_pair("A", "B")
print(f"BSA = {metrics['bsa_A2']} Å²,  SC = {metrics['sc_normalised']}")

```

---

## Output Format

### `interactions_<A><B>.csv`

One row per **(residue pair × interaction type)**; one file per chain pair.  
Only rows for interaction types that were actually detected are written.

| Column | Description |
|---|---|
| `chain_a` | Chain identifier of residue A |
| `residue_a` | One-letter code + sequence number (e.g. `K23`) |
| `dsasa_residue_a` | ΔSASA of residue A upon complex formation (Å²); empty if not at interface |
| `chain_b` | Chain identifier of residue B |
| `residue_b` | One-letter code + sequence number (e.g. `E87`) |
| `dsasa_residue_b` | ΔSASA of residue B (Å²) |
| `interaction_type` | One of: `HBond_donor`, `HBond_acceptor`, `Hydrophobic`, `SaltBridge`, `PiStacking_Par`, `PiStacking_T`, `CationPi`, `Disulfide`, `WaterBridge`, `VanDerWaal` |
| `atoms_a` | Semicolon-separated unique atom names from residue A forming this interaction |
| `atoms_b` | Semicolon-separated unique atom names from residue B forming this interaction |
| `distance` | Minimum contact distance (Å) among all hits of this type for this pair |
| `mediator` | Water residue label(s) for `WaterBridge` (e.g. `HOH401`); empty otherwise |

**Atom assignment invariant**: `atoms_a` always contains atoms from chain A; `atoms_b` always contains atoms from chain B.

**Example**

```
chain_a,residue_a,dsasa_residue_a,chain_b,residue_b,dsasa_residue_b,interaction_type,atoms_a,atoms_b,distance,mediator
A,K23,42.3,B,E87,38.7,SaltBridge,NZ,OE1;OE2,3.812,
A,K23,42.3,B,E87,38.7,HBond_donor,NZ,OE1,3.401,
A,F41,28.1,B,Y95,31.4,PiStacking_Par,ring,ring,4.203,
A,N55,19.6,B,Q102,22.8,HBond_donor,N,OE1,3.147,
A,S31,14.2,B,D74,17.9,WaterBridge,OG,OD1,2.841,HOH401
A,L18,11.5,B,V63,9.2,Hydrophobic,CD1,CG2,3.654,
A,L18,11.5,B,V63,9.2,VanDerWaal,CD1,CG2,3.654,
```

---

### `intra_<chain>.csv`  *(requires `--intra-chain <ID>`)*

One row per **(residue pair × interaction type)** within a single chain.
Same column schema as `interactions_<A><B>.csv`.

Key differences from the intermolecular file:

| Column | Value in intramolecular output |
|---|---|
| `chain_a` | Always equal to `chain_b` (the analysed chain) |
| `dsasa_residue_a` / `dsasa_residue_b` | Always empty (ΔSASA is not applicable within a single chain) |

**Sequence separation** — pairs with fewer than `--intra-min-seq-sep` (default 4)
residues between them are excluded.  This skips bonded neighbours while
capturing biologically relevant secondary-structure interactions:

| Min separation | What is captured |
|---|---|
| 3 | γ-turns (i→i+3) and beyond |
| **4 (default)** | **α-helix H-bonds (i→i+4) and beyond** |
| 5+ | Medium- and long-range contacts only |

**Example**

```
chain_a,residue_a,dsasa_residue_a,chain_b,residue_b,dsasa_residue_b,interaction_type,atoms_a,atoms_b,distance,mediator
A,E5,,A,K9,,HBond_donor,N,O,3.02,
A,A8,,A,L12,,Hydrophobic,CB,CD1,3.81,
A,R15,,A,E22,,SaltBridge,NH1;NH2,OE1;OE2,3.44,
A,F18,,A,Y31,,PiStacking_Par,ring,ring,4.87,
```

---

### `interface_metrics.csv`

One row per chain pair; all pairs in a single file.

| Column | Description |
|---|---|
| `chain_a` | First chain identifier |
| `chain_b` | Second chain identifier |
| `sasa_chain_a_A2` | Total SASA of chain A as a monomer (Å²) |
| `sasa_chain_b_A2` | Total SASA of chain B as a monomer (Å²) |
| `sasa_complex_A2` | Total SASA of the A+B complex (Å²) |
| `bsa_A2` | Buried Surface Area: `(SASA_A + SASA_B − SASA_AB) / 2` (Å²) |
| `sc_normalised` | Shape Complementarity normalised to [0, 1]; empty if too few dot pairs |

---

## Module Reference

### `proteinppi/structure.py` — `StructureLoader`

Parses the input file and exposes shared helpers consumed by every other module.

| Method / attribute | Description |
|---|---|
| `chain_ids` | List of detected protein chain IDs |
| `chain_pairs` | All pairwise combinations of `chain_ids` |
| `waters` | All water residues in the structure |
| `protein_residues(chain_id)` | Standard amino-acid residues for a chain |
| `res_label(residue)` | One-letter + sequence number label (e.g. `"K23"`) |
| `ca_coord(residue)` | Cα coordinate or `None` |
| `dist(a, b)` | Euclidean distance between two coordinate arrays |

---

### `proteinppi/sasa.py` — `SASACalculator`

Wraps **freesasa** (Lee-Richards, 1.4 Å water probe) to compute surface areas.

| Method | Description |
|---|---|
| `compute_for_pair(chain_a, chain_b)` | Returns a dict with total SASA, BSA, per-residue ΔSASA dicts, and interface residue lists |

**BSA** formula:
```
BSA = (SASA_A + SASA_B − SASA_AB) / 2
```

**ΔSASA per residue**: `SASA_monomer(residue) − SASA_complex(residue)`.
Positive values indicate burial at the interface.

---

### `proteinppi/interactions.py` — `InteractionFinder`

Detects all 10 interaction types. Accepts optional ΔSASA dicts from `SASACalculator`
to populate the `dsasa_residue_a / _b` columns in the output.

| Method | Description |
|---|---|
| `find_all(chain_a, chain_b, dsasa_a, dsasa_b, ca_prefilter)` | **Intermolecular** — one record per (residue pair × interaction type) across two chains |
| `find_intra(chain_id, min_seq_sep, ca_prefilter)` | **Intramolecular** — one record per (residue pair × interaction type) within one chain; pairs with sequence-index separation < `min_seq_sep` (default 4) are skipped |

The Cα pre-filter (default 12 Å) is applied in both methods to skip distant pairs for speed.

The module-level constant `INTERACTION_TYPES` lists all valid interaction type strings in canonical order.

---

### `proteinppi/shape_complementarity.py` — `ShapeComplementarity`

Implements the Lawrence & Colman (1993) SC algorithm with normalisation to [0, 1].

| Method | Description |
|---|---|
| `compute(int_res_a, all_res_a, int_res_b, all_res_b)` | Returns normalised SC ∈ [0, 1] or `None` |

**Normalisation**:
```
sc_normalised = clip((sc_raw + 1) / 2, 0, 1)
```

| `sc_normalised` | Interpretation |
|---|---|
| 1.0 | Perfect complementarity |
| ~0.82–0.88 | Typical stable PPI |
| ~0.75 | Crystal contacts |
| 0.5 | Random / uncorrelated surfaces |
| 0.0 | Anti-complementary |

**Key parameters** (set at construction or via CLI):

| Parameter | Default | Effect |
|---|---|---|
| `n_sphere_dots` | 100 | Dots per atom; ↑ accuracy, ↑ runtime |
| `sc_dmax` | 4.0 Å | Max dot-to-dot pairing distance |
| `sc_min_pairs` | 10 | Min pairs to report a score |


---

## Customising Cutoffs

Individual interaction cutoffs can be overridden when calling private finders:

```python
from proteinppi import StructureLoader, InteractionFinder

loader = StructureLoader("1ABC.pdb")
finder = InteractionFinder(loader)

res_a = loader.protein_residues("A")
res_b = loader.protein_residues("B")

# Tighter H-bond cutoff for a specific residue pair
hbonds = finder._find_hbonds(res_a[5], res_b[12], cutoff=3.2)

# Custom Cα pre-filter per pair
records = finder.find_all("A", "B", ca_prefilter=15.0)
```

---

## Citation

If you use ProteinPPI in your research, please cite:

- **Biopython**: Cock et al. (2009) *Bioinformatics* 25(11):1422–1423. https://doi.org/10.1093/bioinformatics/btp163
- **freesasa**: Mitternacht (2016) *F1000Research* 5:189. https://doi.org/10.12688/f1000research.7931.1
- **Shape Complementarity**: Lawrence & Colman (1993) *J. Mol. Biol.* 234:946–950. https://doi.org/10.1006/jmbi.1993.1648
- **NumPy**: Harris et al. (2020) *Nature* 585:357–362. https://doi.org/10.1038/s41586-020-2649-2
- **SciPy**: Virtanen et al. (2020) *Nature Methods* 17:261–272. https://doi.org/10.1038/s41592-019-0686-2


---

## License

MIT License. See [LICENSE](LICENSE) for details.
