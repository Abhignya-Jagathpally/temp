# Drug Structural Features

This document describes the canonical chemical structures pulled for the 11 ResistanceMap target drugs and the recommended way to convert those structures into Morgan fingerprints (ECFP4-style 2048-bit bit vectors) so they can replace the current one-hot drug encoding inside the ResistanceMap pipeline.

The raw, machine-readable per-drug record lives at `data/raw/drug_metadata.json`. It contains canonical and isomeric SMILES from PubChem, the parent ChEMBL SMILES, molecular formula and weight, max clinical phase, first approval year, mechanism of action (action type + textual MoA + ChEMBL target ID + resolved HGNC gene symbol), and 2 PubMed PMIDs per drug for the MoA reference.

## Why this matters

In the current pipeline, each drug appears as one of 11 column indices in `MultiOmicsDataset` (essentially a one-hot ID). The model has no way to know that Bortezomib and Carfilzomib are both proteasome inhibitors with overlapping pharmacophores, or that Panobinostat, Vorinostat, and Romidepsin all share an HDAC-binding warhead. Replacing the one-hot ID with a 2048-bit Morgan fingerprint lets the regressor learn structure-activity relationships, generalize to held-out drugs, and support drug-recommendation queries from chemistry alone. This is the single largest known gap in the v6/v7 model card.

## SMILES table (one row per drug)

| Drug | PubChem CID | ChEMBL ID (parent) | Molecular formula | MW (Da) | Primary target gene | MoA (action type) |
|---|---:|---|---|---:|---|---|
| Bortezomib | 387447 | CHEMBL325041 | C19H25BN4O4 | 384.20 | PSMB5 | INHIBITOR (26S proteasome) |
| Lenalidomide | 216326 | CHEMBL848 | C13H13N3O3 | 259.26 | CRBN | INHIBITOR (CRL4-CRBN E3 ligase) |
| Panobinostat | 6918837 | CHEMBL483254 | C21H23N3O2 | 349.40 | HDAC1 | INHIBITOR (pan-HDAC) |
| Vorinostat | 5311 | CHEMBL98 | C14H20N2O3 | 264.32 | HDAC1 | INHIBITOR (HDAC1/2/3/6) |
| Romidepsin | 5352062 | CHEMBL343448 | C24H36N4O6S2 | 540.70 | HDAC1 | INHIBITOR (pan-HDAC; HDAC1/2 selective) |
| Venetoclax | 49846579 | CHEMBL3137309 | C45H50ClN7O7S | 868.40 | BCL2 | INHIBITOR (Bcl-2) |
| Dinaciclib | 46926350 | CHEMBL2103840 | C21H28N6O2 | 396.50 | CDK9 | INHIBITOR (CDK1/2/5/9) |
| Palbociclib | 5330286 | CHEMBL189963 | C24H29N7O2 | 447.50 | CDK6 | INHIBITOR (CDK4/6) |
| Doxorubicin | 31703 | CHEMBL53463 | C27H29NO11 | 543.50 | TOP2A | INHIBITOR (TOP2A; anthracycline) |
| Etoposide | 36462 | CHEMBL44657 | C29H32O13 | 588.60 | TOP2A | INHIBITOR (TOP2A/TOP2B) |
| Cyclophosphamide | 2907 | CHEMBL88 | C7H15Cl2N2O2P | 261.08 | DNA (no HGNC) | INHIBITOR (alkylating prodrug) |

Full SMILES strings (canonical, isomeric, and ChEMBL parent) are in `data/raw/drug_metadata.json`.

### Notes on salt/parent disambiguation

Four entries had ChEMBL search top-hits that pointed to a salt, hydrate, cocrystal, or prodrug rather than the parent free-base molecule. We deliberately swapped to the parent ChEMBL ID so the SMILES we feed to RDKit will match what PubChem reports as the active species:

- **Bortezomib**: parent CHEMBL325041 (boronic acid). Approved formulation is the D-mannitol cocrystal CHEMBL5315122 (max_phase 4, first_approval 2003); the mechanism record lives on the cocrystal entity. SMILES is the parent.
- **Palbociclib**: top hit was the isethionate salt CHEMBL2364621. Used parent CHEMBL189963 (max_phase 4, first_approval 2015).
- **Etoposide**: top hit was the phosphate prodrug CHEMBL1200645 (approved 1996). Used parent CHEMBL44657 (max_phase 4, first_approval 1983).
- **Cyclophosphamide**: top hit was the monohydrate CHEMBL1200796. Used anhydrous parent CHEMBL88. ChEMBL records the mechanism target as `DNA` (CHEMBL2311221) — not a protein — so `primary_target_gene_symbol` is set to the literal string `"DNA"` and there is no HGNC mapping. Note that cyclophosphamide is itself a prodrug bioactivated by CYP2B6 / CYP3A4 to phosphoramide mustard, which alkylates DNA at N7-guanine.

The JSON sets `smiles_source_disagreement: true` for each of these four records and explains the divergence in the per-drug `notes` field.

## Computing Morgan fingerprints (2048-bit, radius 2 = ECFP4-equivalent)

Five-line snippet using RDKit. Do not install RDKit as part of this PR — install it via `conda install -c conda-forge rdkit` or `pip install rdkit` in a separate environment-setup step.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np, json, torch

def smiles_to_morgan(smi: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.frombuffer(bv.ToBitString().encode(), dtype="u1") - ord("0")

records = json.load(open("data/raw/drug_metadata.json"))
DRUG_ORDER = ["Bortezomib","Lenalidomide","Panobinostat","Vorinostat","Romidepsin",
              "Venetoclax","Dinaciclib","Palbociclib","Doxorubicin","Etoposide","Cyclophosphamide"]
fps = np.stack([smiles_to_morgan(records[d]["isomeric_smiles"]) for d in DRUG_ORDER])  # (11, 2048)
drug_features = torch.from_numpy(fps).float()
```

Use `isomeric_smiles` (it preserves stereochemistry) for the small drugs where chirality matters (Bortezomib, Romidepsin, Doxorubicin, Etoposide). For drugs without stereocenters (Lenalidomide, Vorinostat, Cyclophosphamide, Palbociclib, Venetoclax) `isomeric_smiles` and `canonical_smiles` are identical or near-identical.

## Where this plugs into the ResistanceMap pipeline

Currently in `resistancemap/data/loaders.py`, `MultiOmicsDataset` exposes a drug index but no chemical features. The minimal integration is:

1. **`MultiOmicsDataset.__init__`**: load `data/raw/drug_metadata.json`, compute the 2048-bit Morgan fingerprint matrix once at startup, store it as `self.drug_features: torch.Tensor` of shape `(n_drugs, 2048)` in the canonical drug order above.
2. **`MultiOmicsDataset.__getitem__`**: in addition to the existing `drug_idx`, also return `drug_features[drug_idx]` so each `(sample, drug)` example carries its 2048-bit chemistry vector.
3. **Fusion / regression head**: in the protein/fusion stage of the model, concatenate the per-example drug fingerprint with the protein/omics latent before the final regression. A small projection (`nn.Linear(2048, 64)`) before concatenation keeps the parameter count manageable. This is where a held-out-drug evaluation can finally be meaningful — the model sees chemistry, not an ID.

A future option: replace the precomputed Morgan matrix with a pretrained molecular GNN encoder (see caveat below).

## License and attribution

- **PubChem** data (CID, canonical SMILES, isomeric SMILES, molecular formula, molecular weight) is in the public domain. Cite: Kim S. et al., *PubChem 2023 update*, Nucleic Acids Res. 2023.
- **ChEMBL** data (parent ChEMBL ID, mechanism of action, action type, target ChEMBL ID, gene symbol mapping, max_phase, first_approval) is licensed under **CC BY-SA 3.0**. Cite: Zdrazil B. et al., *The ChEMBL Database in 2023*, Nucleic Acids Res. 2024. Any work that incorporates ChEMBL-derived fields must propagate the CC BY-SA 3.0 license to downstream artifacts.
- **PubMed** PMIDs reference U.S. National Library of Medicine records; the PMID strings themselves are not copyrightable.

## Caveat: Morgan fingerprints capture local chemistry, not 3D shape

ECFP4 / 2048-bit Morgan fingerprints are circular substructure hashes. They encode the local atomic environment around each atom up to radius 2 bonds, which captures functional-group identity, ring systems, and immediate connectivity — but they are explicitly 2D and substructure-based:

- They do **not** encode 3D conformation, stereochemistry beyond what SMILES already specifies, ring puckering, or accessible surface area.
- They do **not** encode binding-site complementarity. Two molecules with very different fingerprints can still bind the same pocket (scaffold hopping); two molecules with similar fingerprints can have wildly different ADMET.
- They are nonetheless an excellent baseline for QSAR-style tasks and are what virtually every published drug-response benchmark (GDSC, CTRP, PRISM) uses as the default chemical featurization.

**Upgrade path**: for a stronger drug encoder, swap Morgan fingerprints for a pretrained molecular GNN — for example a Graph Isomorphism Network (GIN) pretrained on ZINC15 or ChEMBL via contrastive or masked-atom objectives (cf. Hu et al., *Strategies for Pre-training Graph Neural Networks*, ICLR 2020; or MolCLR, Wang et al., 2022). A pretrained GIN encoder produces a 300-dim drug embedding that captures graph-level structure and tends to outperform Morgan on out-of-distribution drug generalization. This swap is plug-compatible with the integration above: `drug_features` simply becomes the GIN output instead of the Morgan bit vector.
