# Stage-1 fusion: LINCS cluster validation

Run date: 2026-11-09. This report describes a new, pooled structure-level benchmark.
It is not a rerun of the historical signature-level notebook.

The observed gains over the best single view are small. Joint L2-PCA gives the
highest mean prediction Pearson; family-weighted L2 concatenation gives the
lowest prediction MSE and highest MoA precision@10. Raw and z-score-based feature
fusion produce catastrophic prediction errors on some held-out structures.
These results support testing L2-based baselines first on this feature collection;
they do not establish a universally superior fusion method.

## Data and preprocessing audit

- Source: LINCS GSE92742 Level-5, 211,810 signatures × 978 landmark genes.
- Valid MoA and target annotations: 65,243 signatures and 2,200 canonical SMILES.
- Shared evaluated cohort: **1,183 structures in 929 compound-name-connected
  groups**, representing 28,730 source signatures. The evaluation unit is a
  structure, not an independently counted compound name or signature.
- Responses are means across the observed cell lines, doses and times, with
  equal weight per signature within each structure. Structures receive equal
  weight within each evaluation fold. This does not test context-specific prediction.
- Eleven views: ChemBERTa-2 MLM (3 × 768), ChemBERTa-2 MTR (3 × 384), MoLFormer
  (3 × 768), regenerated Morgan (radius 2, 1,024 bits), and 217 RDKit descriptors.
  Concatenation therefore has 7,001 features.
- Existing neural embeddings are reused; overlap between their pretraining data
  and downstream entities was not audited.
- The neural CSV reader selects named `e_*` columns. It excludes `row_index`,
  which had incorrectly become a feature in the legacy ChemBERTa AnnData matrices.
- Morgan and RDKit features are regenerated from SMILES because the legacy
  classical CSVs were already globally standardized without saved fit statistics.
  The RDKit version and descriptor names are recorded in the cohort provenance.
- MoLFormer files have conflicting duplicate vectors for 888 CLS, 1,012 max and
  1,013 mean-pooling SMILES. CLS discrepancies reach only 3.34e-6, but max/mean
  discrepancies reach 8.72 and 4.30. The explicit conservative policy excludes
  any conflicting vector; identical duplicates are collapsed. Six structures
  also fail raw descriptor preparation. These exclusion sets overlap.
- In total, 1,017 structures are absent from the shared cohort. Every method uses
  the same remaining structures. This substantial filtering limits generalization;
  all excluded IDs and conflict magnitudes are retained on the cluster.

## Evaluation protocol

Three outer folds × three split seeds (0, 1, 2). Structures connected through
any shared compound name remain together in both outer and inner splits.
Normalization, block scaling and PCA are fitted on the relevant training fold.
The runs compare combinations of preprocessing and fusion; they do not isolate
the contribution of the fusion rule alone.

Regression uses Ridge with alpha in {0.1, 1, 10, 100}, chosen by inner three-fold
CV on negative MSE. There are 11 full-width L2 single-view baselines, seven fusion
configurations, 11 additional L2/PCA64 single-view controls, and a training-mean
response baseline. PCA fusion also uses 64 dimensions. Equal family weights
assign 1/3 to each of the three pooling variants within each neural family, and
1 each to Morgan and RDKit; blocks are multiplied by the square root of weights.

Retrieval compares the 11 single views, seven feature fusions and two late
ensembles. Held-out structures query training structures using cosine similarity.
Relevance is exact equality of the entire MoA annotation string. Precision and
hit rate include queries with no relevant reference. Recall is undefined for
those queries; 854, 848 and 847 of 1,183 queries have a relevant reference in
seeds 0, 1 and 2, respectively.

Numbers below are the unweighted means of nine fold scores. **SD is descriptive
variation across dependent folds/seeds, not a confidence interval.** Configurations
were compared on these folds; a separate final holdout is needed to evaluate a
configuration selected from this comparison. No significance claim is made.

## Prediction results

Pearson is measured across the 978 genes per held-out structure, then averaged.
MSE averages squared prediction errors across genes and structures.

| Configuration | Width | Pearson, mean ± SD | MSE, mean |
| --- | ---: | ---: | ---: |
| Training mean | — | 0.4498 ± 0.0072 | 0.2751 |
| Morgan (L2; best single view) | 1024 | 0.4625 ± 0.0063 | 0.2662 |
| MoLFormer CLS + PCA64 (best matched-width single) | 64 | 0.4609 ± 0.0068 | 0.2698 |
| L2 concatenation | 7001 | 0.4660 ± 0.0081 | 0.2676 |
| L2 concatenation, equal family weights | 7001 | 0.4631 ± 0.0071 | 0.2636 |
| L2 concatenation + joint PCA64 | 64 | 0.4667 ± 0.0074 | 0.2689 |
| Raw concatenation | 7001 | 0.4383 ± 0.0130 | 2.201e+21 |
| Z-score concatenation | 7001 | 0.2758 ± 0.0127 | 1.823e+18 |
| Z-score concatenation, balanced blocks | 7001 | 0.4660 ± 0.0078 | 2.851e+16 |
| Z-score/balanced concatenation + joint PCA64 | 64 | 0.4622 ± 0.0081 | 1.981e+16 |

Joint L2-PCA improves mean Pearson by 0.0042 over the best full-width single view
and by 0.0058 over the best single-view PCA64 control. Family-weighted L2
concatenation has the lowest observed MSE. These are different ranking criteria.

The large MSEs above are retained, not clipped or removed. The diagnostic audit
finds a held-out RDKit `Ipc` value of 4.28e33 versus a maximum of 1.09e22 in one
training fold, producing an absolute test z-score of 1.11e13. Z-scoring does not
bound such extrapolation, and block balancing or PCA does not eliminate it.
Raw concatenation also triggers singular-system warnings in Ridge. A reasonable
correlation can hide severe amplitude errors; those configurations should not
be treated as usable prediction winners. Robust descriptor preprocessing would
need a separately specified, training-fitted protocol and a new benchmark.

## MoA retrieval results

| Configuration | Precision@10, mean ± SD | Hit rate@10, mean |
| --- | ---: | ---: |
| Morgan (L2; best single view) | 0.1401 ± 0.0116 | 0.4067 |
| L2 concatenation, equal family weights | 0.1421 ± 0.0095 | 0.4278 |
| L2 concatenation | 0.1347 ± 0.0075 | 0.4193 |
| Mean cosine score | 0.1347 ± 0.0075 | 0.4193 |
| L2 concatenation + joint PCA64 | 0.1308 ± 0.0089 | 0.4055 |
| Mean rank | 0.1183 ± 0.0071 | 0.3955 |
| Z-score concatenation | 0.1268 ± 0.0105 | 0.4151 |
| Z-score concatenation, balanced blocks | 0.1290 ± 0.0100 | 0.4227 |
| Z-score/balanced concatenation + joint PCA64 | 0.1227 ± 0.0091 | 0.3988 |
| Raw concatenation | 0.0305 ± 0.0034 | 0.1774 |

Family-weighted L2 fusion exceeds Morgan by 0.0019 absolute precision@10 in the
observed means. Uniform L2 concatenation and mean cosine scores have identical
retrieval results here, consistent with the weighted-cosine identity for
nonzero L2-normalized blocks. Mean-rank fusion does not improve on Morgan in
this run. Normalization and fusion are evaluated together; the table does not
isolate their causal contributions.

## Verification and execution

- 81 focused and compatibility tests passed on the cluster. One pre-existing
  morphology-output test was excluded after failing because the cluster's
  PyTorch/torchvision installation lacks `torchvision::nms`; it does not exercise
  fusion. The full repository test suite was not run.
- All seven code cells of the [fusion tutorial](notebooks/07_fusion.ipynb) passed
  in a fresh cluster kernel. Its examples and outputs are explicitly synthetic.
- Result verification checked 162 main regression rows, 99 matched-width rows,
  180 retrieval rows, and nine training-mean baseline rows. It confirmed identical
  group-disjoint outer splits, one out-of-fold record per entity/configuration/seed,
  finite fold metrics, and agreement between saved entity and fold scores.
  Finite scores are not evidence of useful predictions, as the MSE failures show.
- Source hashes confirm that the 11 fusion implementation files in the workspace
  match the isolated cluster snapshot. The active cluster checkout was not overwritten.
- Slurm jobs: cohort preparation `42987570`; test run `42987611`; corrected
  tutorial `42987910`; regression `42987911`; retrieval `42987912`; result audit
  `42988059`. The combined test/tutorial job stopped at an example-axis assertion
  after its tests passed; the corrected tutorial then passed separately.
- Regression completed in 24m17s and retrieval in 2m11s of Slurm elapsed time,
  on CPU node `cpusrv51`, using four CPUs and an 8 GB memory request. Both
  benchmarks used NumPy 2.5.2 and scikit-learn 1.9.0.

All embeddings, prepared targets and entity-level results remain on the cluster.
Only the executed tutorial and small summary/provenance records were retrieved.

```text
Input AnnData:
/lustre/groups/ml01/workspace/goncalo.pinto/embpy-reproducibility/data/lincs_adata_with_embeddings_moa_annotated.h5ad

Embedding root:
/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/embeddings

Source snapshot:
/lustre/groups/ml01/workspace/goncalo.pinto/embpy/.job_tmp/fusion_stage1_20261109/code_v2

Results and audit:
/lustre/groups/ml01/workspace/goncalo.pinto/embpy/runs/fusion_stage1_20261109
```

The result directory contains `regression_comparison.csv`, `retrieval_summary.csv`,
all fold/entity/split tables, `cohort_provenance.json`, `source_manifest.json`,
`execution_*.json`, `verification.json`, logs and the executed tutorial.
The preparation audit records source file hashes, dimensions and exclusions.
The public entry point is `python -m embpy.scripts.benchmark_fusion_lincs`, with
the Slurm template in `submission_scripts/run_fusion_lincs.sbatch`; see the
[fusion guide](fusion.md) for arguments and fit/transform usage.
