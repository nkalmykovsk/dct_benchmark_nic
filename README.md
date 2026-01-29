# DCT Basis Benchmarks for Neural Image Compression: Revealing Frequency Dependent Biases


<p align="center">
  <img src="images/overview_3d_median_leak_allmodels.png" alt="3D Overview of Median Frequency Leakage" width="100%">
</p>

<p align="center">
  <em>3D overview of median frequency leakage (L<sub>k</sub>) for 10 codecs. Each panel: x = quality, y = image size, z = median L<sub>k</sub> (↓ better). Values are medians across frequency bins, averaged over 100 runs.</em>
</p>

---

## Overview

This repository provides a systematic benchmark for evaluating **frequency response characteristics** of image compression codecs using DCT (Discrete Cosine Transform) basis functions as test signals.

**Key Contributions:**
- Novel frequency-domain evaluation framework using DCT basis images
- Comprehensive analysis of 11 codecs: 3 traditional (JPEG, JPEG XL, WebP) and 8 neural (BMShj18-Factorized, BMShj18-Hyperprior, MBT18-Mean, MBT18, Cheng2020-Anchor, Cheng2020-Attn, TCM, FTIC, StableCodec)
- Six frequency-specific metrics revealing codec biases
- Adaptive fine-tuning method to reduce frequency leakage

---

## Evaluated Metrics

We propose six metrics computed from the response matrix **R** obtained by compressing DCT basis images:

| Metric | Symbol | Description |
|--------|--------|-------------|
| **Frequency Leakage** | L<sub>k</sub> | Energy leaking to off-diagonal elements: L<sub>k</sub> = 1 − diag(R)<sub>k</sub> |
| **Off-Diagonal Ratio** | ODR<sub>k</sub> | Ratio of off-diagonal to diagonal response |
| **Centroid Shift** | \|Δc<sub>k</sub>\| | Displacement of energy centroid from diagonal position |
| **Spread** | s<sub>k</sub> | Spatial spread of response around the diagonal |
| **Entropy** | H<sub>k</sub> | Shannon entropy of the normalized response distribution (bits) |
| **Concentration Energy** | CE<sub>k</sub>(w) | Energy ratio within window w around diagonal |

---

## Installation

```bash
git clone https://github.com/nkalmykovsk/dct_benchmark_nic.git
cd dct_benchmark_nic
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, CUDA-capable GPU (recommended)

---

## Usage

### Run Full Benchmark

```bash
jupyter notebook demo.ipynb
```

The notebook executes experiments across:
- **Models:** JPEG, JPEG XL, WebP, BMShj18-Factorized, BMShj18-Hyperprior, MBT18-Mean, MBT18, Cheng2020-Anchor, Cheng2020-Attn, TCM, FTIC, StableCodec
- **Quality levels:** q ∈ {1, 2, 3, 4, 5, 6} (p ∈ {64, 128} for TCM)
- **Image sizes:** 64×64, 128×128, 256×256, 512×512, 1024×1024 (256×256+ for TCM, FTIC, StableCodec)

### Generate Figures

```bash
python3 utils/generate_figures.py --root results/
```

---

## Project Structure

```
dct_benchmark_nic/
├── demo.ipynb              # Main experiment notebook
├── utils/
│   ├── functions.py        # Core metrics and evaluation functions
│   ├── loaders.py          # Model loading utilities
│   ├── generate_figures.py # Paper figure generation
│   └── tcm_setup.py        # TCM model setup
├── results/                # Experiment outputs (metrics, images)
├── images/                 # Figures for README
└── requirements.txt
```

## Citation

```bibtex
@inproceedings{kalmykov2025dct,
  title={DCT Basis Benchmarks for Neural Image Compression: Revealing Frequency Dependent Biases},
  author={Kalmykov, Nikolay and ...},
  booktitle={Proceedings of [Conference Name]},
  year={2025}
}
```

---

## License

This project is released under the **MIT License**. See `LICENSE`.

---

## Model Setup

### Neural Codecs

Most neural codecs (CompressAI models) are automatically downloaded. For TCM, FTIC, and StableCodec, follow these steps:

#### TCM (Token-based Context Model)
See `utils/tcm_setup.py` for automatic setup.

#### FTIC (Frequency-aware Transformer for Image Compression, ICLR 2024)
1. Clone the repository:
   ```bash
   cd third_party
   git clone https://github.com/xyq7/ICLR2024-FTIC.git
   ```
2. Download checkpoints from the repo's Google Drive link into `third_party/ICLR2024-FTIC/checkpoints/`

#### StableCodec (One-Step Diffusion for Extreme Compression, ICCV 2025)

1. **Clone the repository:**
   ```bash
   cd third_party
   git clone https://github.com/LuizScarlet/StableCodec.git
   ```

2. **Download SD-Turbo (base diffusion model):**
   ```bash
   # Using huggingface-cli (recommended)
   pip install huggingface-cli
   huggingface-cli download stabilityai/sd-turbo --local-dir third_party/sd-turbo
   
   # Or using git-lfs
   cd third_party
   git clone https://huggingface.co/stabilityai/sd-turbo
   ```

3. **Download StableCodec checkpoints:**
   
   Visit [Google Drive folder](https://drive.google.com/drive/folders/1itiVVAPSTATGPcHLp_bLI9r9Qi3YcM12?usp=sharing) and download:
   - `elic_official.pth` (auxiliary encoder, ~155 MB)
   - `stablecodec_ft2.pkl` (~0.035 bpp, quality 1)
   - `stablecodec_ft4.pkl` (~0.025 bpp, quality 2)
   - `stablecodec_ft8.pkl` (~0.017 bpp, quality 3)
   - `stablecodec_ft12.pkl` (~0.013 bpp, quality 4)
   - `stablecodec_ft16.pkl` (~0.010 bpp, quality 5)
   - `stablecodec_ft32.pkl` (~0.005 bpp, quality 6)
   
   Place all files in: `third_party/StableCodec/checkpoints/`
   
   Using `gdown`:
   ```bash
   mkdir -p third_party/StableCodec/checkpoints
   cd third_party/StableCodec/checkpoints
   gdown --folder 1itiVVAPSTATGPcHLp_bLI9r9Qi3YcM12 --remaining-ok
   ```

4. **Install additional dependencies:**
   ```bash
   pip install diffusers accelerate transformers peft
   ```

**Note:** StableCodec is a diffusion-based codec designed for extreme low bitrates (0.005-0.035 bpp) with one-step denoising. It requires more memory and compute than traditional codecs but produces highly realistic reconstructions.

---

## Acknowledgments

- [CompressAI](https://github.com/InterDigitalInc/CompressAI) for neural codec implementations
- [LIC_TCM](https://github.com/jmliu206/LIC_TCM) for the TCM model
- [ICLR2024-FTIC](https://github.com/xyq7/ICLR2024-FTIC) for the FTIC model
- [StableCodec](https://github.com/LuizScarlet/StableCodec) for the StableCodec model (ICCV 2025)
