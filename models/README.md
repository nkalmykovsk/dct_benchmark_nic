# Fine-tuned models

Large checkpoints are distributed as **GitHub Release assets**, not tracked in git.

## Leakage-fine-tuned decoder (R1.4)

`decoder_g_s_cheng2020-anchor_q6.pth` — the decoder (`g_s`) of Cheng2020-Anchor (q=6)
after joint decoder-only fine-tuning with the leakage loss (`L = MSE(natural patches)
+ λ·L_leak(DCT basis)`, encoder/entropy frozen → bitrate unchanged).

**Download:** see the latest [GitHub Release](https://github.com/nkalmykovsk/dct_benchmark_nic/releases).

**Use:**
```bash
# place the file here, then reproduce the natural-image figure:
mkdir -p results/finetune_natural_heldout
mv decoder_g_s_cheng2020-anchor_q6.pth results/finetune_natural_heldout/
python scripts/make_ft_natural_figure.py --annotate --zoom
```

Regenerate from scratch instead with `scripts/run_finetune_natural.py`.
