# References

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
