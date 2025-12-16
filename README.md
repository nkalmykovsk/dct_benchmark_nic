# DCT Basis Benchmarks for Neural Image Compression: Revealing Frequency Dependent Biases

## 3D Overview of Median Frequency Leakage

![3D overview of median frequency leakage (L_k) for 10 codecs](images/overview_3d_median_leak_allmodels.png)

**Figure Caption:** 3D overview of median frequency leakage (L_k) for 10 codecs (left to right: JPEG, JPEG XL, WebP, BMShj18-Fact, BMShj18-Hyper, MBT18-Mean, MBT18, Cheng2020-Anchor, Cheng2020-Attn, TCM). In each panel: x=quality q (TCM uses p), y=image size (pixels), z=median L_k (lower is better). Metrics are computed on a DCT-basis input; for each configuration (model, size, quality) the value is the median across frequency bins k, averaged over 100 runs. Line color/marker encodes the size (64–1024).