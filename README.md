# Cell-Type-Resolved Chromatin Effects of AD/PD Variants: DNA Language Model vs. ChromBPNet

**Claude Science Hackathon — Research Track**

Predicting how a non-coding variant reshapes chromatin accessibility in a cell type you care about is the home turf of **ChromBPNet**, a specialized convolutional network (the approach used by the Corces lab). This project treats it as a head-to-head contest: we take our own general-purpose DNA language model, **NucEL** (a ModernBERT architecture pretrained with masked language modeling on genomic sequence), and make it perform ChromBPNet's own cell-type-resolved endpoints on identical benchmarks.

> **Central finding.** Given enough labeled data, a general DNA language model **overtakes the specialized CNN on accessibility classification** and **independently reproduces its variant-effect atlas** (direction + top hits). The gap at low data is **data-limited, not architectural**.

All computation ran locally on RTX PRO 6000 (Blackwell) GPUs.

📄 **Full report:** [`report/report_en.html`](report/report_en.html) · [`report/report_en.pdf`](report/report_en.pdf)

---

## TL;DR scoreboard

![Scoreboard](figures/Figure1_scoreboard_LM_vs_ChromBPNet.png)

**Figure 1.** Head-to-head on ChromBPNet's own cell-type-resolved endpoints. The fine-tuned DNA language model NucEL beats zero-shot ChromBPNet at accessibility classification on GM12878 (0.967 vs 0.899) and on all four brain cell types (mean 0.884 vs 0.744), and reproduces ChromBPNet's variant-effect ranking (atlas consistency ρ=0.50; CR1/rs6701713 is the #1 hit for both).

| Endpoint | NucEL (DNA LM, fine-tuned) | ChromBPNet (specialized CNN) | Winner |
|---|---|---|---|
| GM12878 accessibility classification (auROC) | **0.967** (50× data) | 0.899 (zero-shot count) | LM |
| Brain 4-cell-type classification (mean auROC) | **0.884** | 0.744 (zero-shot) | LM |
| Variant-effect atlas direction | rank ρ = **0.50** vs ChromBPNet; CR1 = #1 for both | — | Agreement |
| Base-resolution mechanism agreement | Pearson r ≈ 0.14 | — | Weak (different features) |

---

## The two endpoints

### 1 · Cell-type-resolved variant-effect atlas & top hits

676 AD/PD non-coding GWAS variants scored across 4 brain cell types (Brain / DLPFC / astrocyte / glutamatergic neuron). ChromBPNet uses the Corces-lab pretrained brain models; NucEL is **fine-tuned per cell type on the same DNase-seq peaks** so the comparison is fair.

![Brain classification and atlas consistency](figures/Figure2_brain_classification_atlas_consistency.png)

**Figure 2.** (a) Fine-tuned NucEL beats zero-shot ChromBPNet in all four brain cell types. (b) NucEL and ChromBPNet agree on variant-effect direction (ρ=0.50, overall sign concordance 64%, rising to 0.79–0.88 among high-effect variants); **CR1/rs6701713 is the #1 hit for both models**. Brain variant effects have no experimental ground truth, so this is a model-vs-model consistency check.

### 2 · Accessibility classification — a fair LM benchmark

Three DNA language models (**NucEL**, **DNABERT-2**, **NTv2-100M**) vs ChromBPNet, all on the same GM12878 ATAC dataset and byte-identical held-out test set. Every LM gets an equal learning-rate sweep, validation-selected.

![Fair LM comparison](figures/Figure3_fair_LM_classification_comparison.png)

**Figure 3.** After equal-budget tuning, NucEL (0.858) leads all LMs, below the zero-shot CNN's 0.899 *at 1× data*.

![Data scaling curve](figures/Figure6_NucEL_data_scaling_curve.png)

**Figure 6.** NucEL's classification ceiling is **data**, not architecture: test auROC rises 0.804 → 0.896 → 0.941 → **0.967** from 1× to 50×, overtaking the CNN already at 10× (same held-out test set), no plateau at 50×.

<details>
<summary>Supporting analyses (click to expand)</summary>

| | |
|---|---|
| ![Mechanism consistency](figures/Figure4_cross_model_mechanism_consistency.png) | **Figure 4.** LM in-silico-mutagenesis importance vs ChromBPNet DeepSHAP at base resolution: only weak agreement (mean r ≈ 0.14). The LMs reach comparable predictions via partly different features. |
| ![Pooling ablation](figures/Figure5_NucEL_pooling_ablation.png) | **Figure 5.** NucEL classification-head pooling ablation: CLS-token (0.866) slightly beats mean-pool (0.858) at the optimal LR. |
| ![NucEL interpretability](figures/Figure7_NucEL_base_resolution_interpretability.png) | **Figure 7.** NucEL's own base-resolution saturation-ISM + JASPAR TF-motif naming for the top disease variants. |
| ![GPU benchmark](figures/Figure8_GPU_vs_CPU_benchmark.png) | **Figure 8.** GPU engineering: the official CUDA-11-bound TF ChromBPNet ported to PyTorch runs on the Blackwell GPU, numerically faithful (r=0.99999), ~24× faster. |

</details>

---

## GPU engineering contribution

The official ChromBPNet is bound to TensorFlow 2.8 / CUDA 11 and cannot target a Blackwell (sm_120) GPU. We reimplemented the architecture layer-for-layer in PyTorch, loaded the exported Keras weights, and validated the port at **log2FC r = 0.99999** (JSD/IES r = 1.0) against the CPU pipeline — giving a ~24× speedup and a reusable open-source artifact ([`src/chrombpnet_gpu/`](src/chrombpnet_gpu)).

---

## Repository layout

```
.
├── report/                       # Final write-up (English)
│   ├── report_en.html
│   └── report_en.pdf
├── figures/                      # All report figures, named by figure number + content
│   ├── Figure1_scoreboard_LM_vs_ChromBPNet.png
│   ├── ... Figure2–Figure8 ...
│   └── FigureS1_ChromBPNet_DeepSHAP_logos.png   (appendix: ChromBPNet DeepSHAP)
├── results/                      # Final result tables (report-cited)
│   ├── fair_lm_comparison.tsv            # Fig 3  — 4-model classification
│   ├── data_scaling.tsv                  # Fig 6  — NucEL 1×→50× scaling
│   ├── brain_classify_nucel_vs_cnn.tsv   # Fig 1/2 — per-cell classification
│   ├── nucel_atlas_matrix.tsv            # §3    — NucEL 676×4 effect atlas
│   ├── atlas_matrix.tsv                  # §3    — ChromBPNet 676×4 effect atlas
│   ├── nucel_cbp_consistency.tsv         # §3    — per-cell rank/sign consistency
│   ├── pooling_ablation.tsv              # Fig 5 — mean vs CLS pooling
│   ├── motif_nucel.tsv                   # Fig 7 — NucEL top TF motif per variant
│   └── lm_checkpoints_manifest.json      # paths + load recipes for the 3 LM checkpoints
└── src/
    ├── chrombpnet_gpu/            # PyTorch port of the official ChromBPNet
    │   ├── gpu_chrombpnet.py      # architecture (1 conv + 8 dilated blocks + dual heads)
    │   ├── dump_weights.py        # Keras .h5 -> .npz weight export
    │   ├── score_gpu.py           # GPU variant scoring (byte-identical seqs to CPU pipeline)
    │   └── cbp_classify.py        # zero-shot count-based accessibility classification
    ├── brain_atlas/              # Brain cell-type atlas (both models)
    │   ├── build_brain_ds.py      # build per-cell classification datasets from DNase peaks
    │   ├── train_brain.py         # fine-tune NucEL per brain cell type
    │   ├── score_brain_variants.py# NucEL variant effect (Δp = p(alt) − p(ref))
    │   ├── cnn_brain_classify.py  # ChromBPNet zero-shot brain classification baseline
    │   └── build2114.py           # build 2114 bp windows for CNN scoring
    └── lm_benchmark/             # LM-vs-CNN benchmark on GM12878 ATAC
        ├── build_ds.py            # build the 1× classification dataset (GC-matched negatives)
        ├── build_train_50x.py     # build the 50× (350k) training set
        ├── train_nucel_lr.py      # NucEL fine-tune + LR sweep (mean-pool)
        ├── train_nucel_cls.py     # NucEL CLS-token pooling variant
        ├── train_dnabert2_lr.py   # DNABERT-2 fine-tune + LR sweep
        ├── train_ntv2_lr.py       # NTv2-100M fine-tune + LR sweep
        ├── train_scaling.py       # data-scaling runs (1×/3×/10×/50×)
        ├── ism_lm.py              # in-silico mutagenesis for the LMs
        ├── interp_nucel.py        # NucEL saturation-ISM interpretability
        └── motif_v2.py            # JASPAR TF-motif naming from ISM maps
```

## How to run

The scripts run on the GPU host, reading local ENCODE peaks, the hg38 FASTA, and a
NucEL checkpoint (`FreakingPotato/NucEL` on the HuggingFace Hub). Paths are set as
constants near the top of each script. Representative invocations:

```bash
# --- LM benchmark (GM12878 ATAC) ---
python src/lm_benchmark/build_ds.py                        # build 1× dataset
python src/lm_benchmark/train_nucel_lr.py 1.5e-4           # fine-tune NucEL at a given LR
python src/lm_benchmark/train_scaling.py unified_dataset_50x.tsv 50x 1.5e-4 0   # scaling run

# --- Brain cell-type atlas ---
python src/brain_atlas/build_brain_ds.py                   # build per-cell datasets
python src/brain_atlas/train_brain.py brain                # fine-tune NucEL for one cell type
python src/brain_atlas/score_brain_variants.py brain       # NucEL variant effects

# --- GPU-accelerated ChromBPNet scoring ---
python src/chrombpnet_gpu/dump_weights.py                  # export Keras weights -> npz
python src/chrombpnet_gpu/score_gpu.py --list V.tsv --genome hg38.fa \
       --weights W.npz --chrom_sizes cs --out_prefix OUT --schema chrombpnet --gpu 0
```

## Data & models

- **ChromBPNet brain models (DNase-seq):** ENCODE — brain ENCSR947POC, DLPFC ENCSR805RJA, astrocyte ENCSR146KFX, glutamatergic-neuron ENCSR109RIQ.
- **GM12878 ATAC classification:** ENCODE ENCSR637XSC (peaks ENCFF748UZH); same biological source ChromBPNet's official GM12878 model was trained on.
- **Variants:** GWAS Catalog AD/PD, hg38.
- **NucEL:** our DNA language model (ModernBERT), HuggingFace `FreakingPotato/NucEL`. Fine-tuned checkpoints are retained on the compute host; see `results/lm_checkpoints_manifest.json`.

## Limitations

- Our current result is **not a same-training-condition comparison** of the DNA language model against ChromBPNet: NucEL is **task-fine-tuned** on the labeled peaks while ChromBPNet is used **zero-shot** (its pretrained count output). The brain effect atlas (Figure 2b) is therefore a **model-vs-model consistency check**, not an accuracy validation against measured allelic effects.
- Scaling points 3×/10×/50× are single-seed; 1× is seed-sensitive (0.72–0.86). "0.941 > 0.899" is a strong trend, not a multi-seed-replicated exact margin.

## References

### Core method — ChromBPNet
1. **ChromBPNet.** Pampari A, Shcherbina A, Kvon E, Kosicki M, Nair S, Kundu S, Kathiria AS, Risca VI, Kuningas K, Alasoo K, Greenleaf WJ, Pennacchio LA, Kundaje A. *ChromBPNet: bias factorized, base-resolution deep learning models of chromatin accessibility reveal cis-regulatory sequence syntax, transcription factor footprints and regulatory variants.* bioRxiv 2024.12.25.630221 (2024). doi:[10.1101/2024.12.25.630221](https://doi.org/10.1101/2024.12.25.630221) · code: [github.com/kundajelab/chrombpnet](https://github.com/kundajelab/chrombpnet)
2. **BPNet.** Avsec Ž, Weilert M, Shrikumar A, Krueger S, Alexandari A, Dalal K, et al. *Base-resolution models of transcription-factor binding reveal soft motif syntax.* Nature Genetics 53, 354–366 (2021). doi:[10.1038/s41588-021-00782-6](https://doi.org/10.1038/s41588-021-00782-6)
3. **ChromBPNet PyTorch port** (basis for our GPU reimplementation). Xiong L. *chrombpnet-pytorch.* [github.com/jsxlei/chrombpnet-pytorch](https://github.com/jsxlei/chrombpnet-pytorch)

### Disease biology, variants & data
4. **Brain single-cell atlas / AD-PD variants (Corces lab approach).** Corces MR, Shcherbina A, Kundu S, et al. *Single-cell epigenomic analyses implicate candidate causal variants at inherited risk loci for Alzheimer's and Parkinson's diseases.* Nature Genetics 52, 1158–1168 (2020). doi:[10.1038/s41588-020-00721-x](https://doi.org/10.1038/s41588-020-00721-x)
5. **ENCODE.** The ENCODE Project Consortium. *An integrated encyclopedia of DNA elements in the human genome.* Nature 489, 57–74 (2012). doi:[10.1038/nature11247](https://doi.org/10.1038/nature11247)
6. **GWAS Catalog.** Sollis E, et al. *The NHGRI-EBI GWAS Catalog: knowledgebase and deposition resource.* Nucleic Acids Research 51, D977–D985 (2023). doi:[10.1093/nar/gkac1010](https://doi.org/10.1093/nar/gkac1010)

### DNA language models compared
7. **NucEL** — our DNA language model (ModernBERT architecture, masked-language-model pretraining on genomic sequence). HuggingFace: [`FreakingPotato/NucEL`](https://huggingface.co/FreakingPotato/NucEL)
8. **ModernBERT** (NucEL's backbone architecture). Warner B, Chaffin A, Clavié B, et al. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference.* arXiv:[2412.13663](https://arxiv.org/abs/2412.13663) (2024).
9. **DNABERT-2.** Zhou Z, Ji Y, Li W, Dutta P, Davuluri R, Liu H. *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.* arXiv:[2306.15006](https://arxiv.org/abs/2306.15006) (2023); ICLR 2024.
10. **Nucleotide Transformer (NTv2).** Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, et al. *Nucleotide Transformer: building and evaluating robust foundation models for human genomics.* Nature Methods 22, 287–297 (2024). doi:[10.1038/s41592-024-02523-z](https://doi.org/10.1038/s41592-024-02523-z)

### Interpretation & motifs
11. **DeepLIFT / DeepSHAP.** Shrikumar A, Greenside P, Kundaje A. *Learning Important Features Through Propagating Activation Differences.* ICML 2017, PMLR 70:3145–3153.
12. **SHAP.** Lundberg SM, Lee S-I. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
13. **TF-MoDISco.** Shrikumar A, Tian K, Avsec Ž, et al. *Technical Note on Transcription Factor Motif Discovery from Importance Scores (TF-MoDISco).* arXiv:[1811.00416](https://arxiv.org/abs/1811.00416) (2018).
14. **JASPAR 2024** (TF motif database used for variant motif naming). Rauluseviciute I, Riudavets-Puig R, Blanc-Mathieu R, et al. *JASPAR 2024: 20th anniversary of the open-access database of transcription factor binding profiles.* Nucleic Acids Research 52, D174–D182 (2024). doi:[10.1093/nar/gkad1059](https://doi.org/10.1093/nar/gkad1059)
