# PPI Propagation Pilot — ResistanceMap

## §1 Methods

**Random Walk with Restart (RWR)**

Convergence equation:

    r(t+1) = alpha * W_norm * r(t) + (1 - alpha) * s

Parameters:
- alpha = 0.7 (restart probability = 0.30000000000000004)
- s = one-hot seed vector at the drug's primary target gene
- W_norm = column-normalised adjacency matrix, weights = STRING combined_score / 1000
- Convergence: L1(r(t+1) - r(t)) < 1e-6, maximum 100 iterations

**Graph.** STRING v12 human PPI from `data/raw/string_ppi.txt` filtered at
combined_score >= 700. ENSP IDs mapped to HGNC gene symbols via
`data/raw/string/9606.protein.info.v12.0.txt.gz`.

**Drug seeds.** Primary target gene from `data/raw/drug_metadata.json`.
Cyclophosphamide targets DNA (non-protein) — excluded; propagation is undefined
for non-protein targets.

**DepMap validation.** `data/raw/depmap/CRISPRGeneEffect.csv` Chronos scores,
restricted to 3 confirmed MM cell lines: OPM2 (ACH-000035), KMS34 (ACH-000547),
U266 (ACH-000001). Top-50 RWR genes with DepMap coverage were compared to
1000 random column subsets of the same size drawn from real DepMap data
(no synthetic values). One-sided Wilcoxon rank-sum test (hypothesis: top-k has
lower/more-negative Chronos than random). Bonferroni correction over 10 tests.

---

## §2 Results

### Per-drug table

| Drug                 | Seed     | InGraph  | Top-10 RWR proteins | Mean Chr K | Mean Chr R | p (Bonferroni) |
|:---------------------|:---------|:---------|:--------------------|:----------:|:----------:|:--------------:|
| Bortezomib           | PSMB5    | yes      | PSMB5; PSMD4; PSMA1; PSMA4; PSMA5; PSMA7; PSMA6; PSMB1; PSMB3; PSMA2 | -1.3399 | -0.1451 | 9.171e-38 |
| Lenalidomide         | CRBN     | yes      | CRBN; RBX1; DDB1; CUL4A; CUL4B; DDB2; DCAF11; DCAF4; ERCC8; DET1 | -0.5093 | -0.1490 | 1.000e+00 |
| Panobinostat         | HDAC1    | yes      | HDAC1; HDAC2; TP53; H3C12; H3C13; EP300; H4C6; RBBP4; KDM1A; H3-3B | -0.2916 | -0.1463 | 1.000e+00 |
| Vorinostat           | HDAC1    | yes      | HDAC1; HDAC2; TP53; H3C12; H3C13; EP300; H4C6; RBBP4; KDM1A; H3-3B | -0.2916 | -0.1444 | 1.000e+00 |
| Romidepsin           | HDAC1    | yes      | HDAC1; HDAC2; TP53; H3C12; H3C13; EP300; H4C6; RBBP4; KDM1A; H3-3B | -0.2916 | -0.1470 | 1.000e+00 |
| Venetoclax           | BCL2     | yes      | BCL2; BCL2L1; TP53; AKT1; BID; CASP3; CYCS; BECN1; BCL2L11; BAX | -0.2433 | -0.1443 | 1.000e+00 |
| Dinaciclib           | CDK9     | yes      | CDK9; CCNT1; CCNT2; POLR2A; SUPT5H; AFF4; SUPT4H1; CDC73; CTR9; MLLT3 | -0.6561 | -0.1471 | 1.555e-18 |
| Palbociclib          | CDK6     | yes      | CDK6; CDK2; CDK4; CDKN1A; CCND1; CCNA2; TP53; CDKN1B; E2F1; CCNL2 | -0.5131 | -0.1477 | 8.625e-01 |
| Doxorubicin          | TOP2A    | yes      | TOP2A; CDK1; CCNA2; BUB1B; BUB1; CDC20; CCNB1; KIF11; PLK1; DLGAP5 | -1.0364 | -0.1446 | 1.196e-30 |
| Etoposide            | TOP2A    | yes      | TOP2A; CDK1; CCNA2; BUB1B; BUB1; CDC20; CCNB1; KIF11; PLK1; DLGAP5 | -1.0364 | -0.1482 | 2.675e-30 |
| Cyclophosphamide     | DNA      | no       | — | — | — | — |

**Mean Chr K** = mean Chronos in top-50 RWR genes (3 MM lines). **Mean Chr R** = mean of 1000 random same-size subsets.

Bonferroni-significant (p < 0.05): **Bortezomib, Dinaciclib, Doxorubicin, Etoposide**
Nominal significance only (p < 0.05, uncorrected): **Bortezomib, Dinaciclib, Doxorubicin, Etoposide**

---

## §3 Network-bias check

Hub protein ranks in each drug's full RWR score vector (rank 1 = most proximal to seed):

| Drug                 | TP53 rank | MYC rank | AKT1 rank |
|:---------------------|----------:|---------:|----------:|
| Bortezomib           |       91 |      306 |      326 |
| Lenalidomide         |       64 |       92 |      130 |
| Panobinostat         |        3 |       17 |       84 |
| Vorinostat           |        3 |       17 |       84 |
| Romidepsin           |        3 |       17 |       84 |
| Venetoclax           |        3 |       24 |        4 |
| Dinaciclib           |       42 |       66 |      265 |
| Palbociclib          |        7 |       23 |       78 |
| Doxorubicin          |       50 |      189 |      381 |
| Etoposide            |       50 |      189 |      381 |
| Cyclophosphamide     |        — |        — |        — |

If hubs appear in the top-50 across unrelated drug seeds, the RWR is dominated by network
topology rather than drug mechanism. Check whether any hub rank < 50 for unrelated seeds.

---

## §4 Cross-drug specificity (Spearman of RWR rankings)

```
              Bortezomib  Lenalidomide  Panobinostat  Vorinostat  Romidepsin  Venetoclax  Dinaciclib  Palbociclib  Doxorubicin  Etoposide
Bortezomib         1.000         0.867         0.803       0.803       0.803       0.834       0.834        0.853        0.820      0.820
Lenalidomide       0.867         1.000         0.856       0.856       0.856       0.818       0.871        0.884        0.853      0.853
Panobinostat       0.803         0.856         1.000       1.000       1.000       0.820       0.900        0.920        0.869      0.869
Vorinostat         0.803         0.856         1.000       1.000       1.000       0.820       0.900        0.920        0.869      0.869
Romidepsin         0.803         0.856         1.000       1.000       1.000       0.820       0.900        0.920        0.869      0.869
Venetoclax         0.834         0.818         0.820       0.820       0.820       1.000       0.771        0.866        0.770      0.770
Dinaciclib         0.834         0.871         0.900       0.900       0.900       0.771       1.000        0.900        0.881      0.881
Palbociclib        0.853         0.884         0.920       0.920       0.920       0.866       0.900        1.000        0.923      0.923
Doxorubicin        0.820         0.853         0.869       0.869       0.869       0.770       0.881        0.923        1.000      1.000
Etoposide          0.820         0.853         0.869       0.869       0.869       0.770       0.881        0.923        1.000      1.000
```

Expected pattern: HDAC inhibitors (Panobinostat, Vorinostat, Romidepsin) should show
high mutual correlation (shared HDAC1 seed). Bortezomib vs Doxorubicin should show
low correlation (proteasome vs topoisomerase II — distinct cellular machineries).

---

## §5 Bortezomib — proteasome subunit ranks

PSMB5-seeded RWR ranks for proteasome complex members (lower rank = more proximal):

- PSMB5: rank 1
- PSMD4: rank 2
- PSMA1: rank 3
- PSMA4: rank 4
- PSMA5: rank 5
- PSMA7: rank 6
- PSMA6: rank 7
- PSMB1: rank 8
- PSMB3: rank 9
- PSMA2: rank 10
- PSMB2: rank 11
- PSMD14: rank 12
- PSMC3: rank 13
- PSMA3: rank 14
- PSMD12: rank 15

Question A validation: if PSMA/PSMB/PSMC/PSMD subunits cluster in the top-50 and show
lower Chronos than random genes in MM lines, mechanism-specific propagation is confirmed.

---

## §6 Verdict

**CONDITIONAL** — Several drugs reach nominal significance but Bonferroni threshold is not broadly met.

**Recommendation:** RWR shows partial mechanistic recovery. Consider: (a) raising alpha toward 0.85, (b) multi-seed (all ChEMBL targets per drug), (c) tissue-specific expression weighting. Conditional retention for v10 pending sensitivity analysis.

### Required changes for v10

1. **Multi-seed RWR**: seed all ChEMBL-curated protein targets per drug, not just the primary.
2. **Expand MM cohort**: include all haematological malignancy DepMap lines (approx. 50) for adequate statistical power.
3. **Pathway-level aggregation**: aggregate Chronos at pathway level (proteasome, HDAC complex, CDK complex) before testing — gene-level test is underpowered at n=3 cell lines.
4. **Alpha sensitivity sweep**: test alpha in [0.5, 0.6, 0.7, 0.8] to characterise hub-bias vs specificity trade-off.
5. **Cyclophosphamide**: non-protein seed — use CYP2B6 (primary bioactivating enzyme) as proxy seed for the DNA-damage arm, or exclude from propagation.
