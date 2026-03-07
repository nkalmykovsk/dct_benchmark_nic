"""Model loading utilities for NIC frequency response evaluation."""

import os
import sys
import warnings
from io import BytesIO
from pathlib import Path

warnings.filterwarnings("ignore", message=".*IProgress not found.*")

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

try:
    import imagecodecs as _ic
except ImportError:
    _ic = None


class ImageCodecModel:
    """Wrapper for traditional image codecs (JPEG, WebP, JPEG2000, JPEG XL)."""

    SUPPORTED_CODECS = {'jpeg', 'webp', 'jpeg2000', 'jpegxl'}

    def __init__(self, codec: str, q_level: int = 1,
                 jpeg_quality: int | None = None,
                 webp_quality: int | None = None,
                 jpegxl_distance: float | None = None):
        self.codec = codec.lower()
        if self.codec not in self.SUPPORTED_CODECS:
            raise ValueError(
                f"Unsupported codec: {codec}. Use: {self.SUPPORTED_CODECS}"
            )
        self.q_level = int(q_level)
        self._jpeg_quality = jpeg_quality
        self._webp_quality = webp_quality
        self._jpegxl_distance = jpegxl_distance

    @staticmethod
    def _map_q_to_quality(q: int, max_q: int = 6) -> int:
        """Map abstract quality level (1..max_q) to JPEG/WebP scale (20..95)."""
        q = max(1, min(int(q), max_q))
        return int(np.linspace(20, 95, num=max_q)[q - 1])

    def __call__(self, x: torch.Tensor) -> dict:
        """Compress and decompress input tensor, return dict with 'x_hat' and 'bpp'."""
        img_u8 = self._tensor_to_uint8(x)
        out_u8, nbytes = self._encode_decode(img_u8)
        x_hat = self._uint8_to_tensor(out_u8)
        h, w = img_u8.shape[0], img_u8.shape[1]
        bpp = 8.0 * nbytes / (h * w) if (h * w) > 0 else 0.0
        return {"x_hat": x_hat, "bpp": bpp}

    def _tensor_to_uint8(self, x: torch.Tensor) -> np.ndarray:
        """Convert [1,3,H,W] tensor in [0,1] to uint8 HxWx3 array."""
        img = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def _uint8_to_tensor(self, out_u8: np.ndarray) -> torch.Tensor:
        """Convert uint8 HxWx3 array to [1,3,H,W] tensor in [0,1]."""
        if out_u8.ndim == 2:
            out_u8 = np.stack([out_u8] * 3, axis=-1)
        out_f = out_u8.astype(np.float32) / 255.0
        return torch.from_numpy(out_f).permute(2, 0, 1).unsqueeze(0)

    def _encode_decode(self, img_u8: np.ndarray) -> tuple[np.ndarray, int]:
        """Encode and decode image. Returns (decoded_image, nbytes_compressed)."""
        if self.codec == 'jpeg':
            return self._jpeg_roundtrip(img_u8)
        elif self.codec == 'webp':
            return self._webp_roundtrip(img_u8)
        elif self.codec == 'jpeg2000':
            return self._jpeg2000_roundtrip(img_u8)
        elif self.codec == 'jpegxl':
            return self._jpegxl_roundtrip(img_u8)
        raise ValueError(f"Unsupported codec: {self.codec}")

    def _jpeg_roundtrip(self, img_u8: np.ndarray) -> tuple[np.ndarray, int]:
        q_val = self._jpeg_quality if self._jpeg_quality is not None else self._map_q_to_quality(self.q_level)
        buf = BytesIO()
        Image.fromarray(img_u8, mode='RGB').save(
            buf, format='JPEG', quality=q_val, subsampling=0, optimize=True
        )
        nbytes = buf.getbuffer().nbytes
        buf.seek(0)
        return np.array(Image.open(buf).convert('RGB')), nbytes

    def _webp_roundtrip(self, img_u8: np.ndarray) -> tuple[np.ndarray, int]:
        q_val = self._webp_quality if self._webp_quality is not None else self._map_q_to_quality(self.q_level)
        buf = BytesIO()
        Image.fromarray(img_u8, mode='RGB').save(
            buf, format='WEBP', quality=q_val, method=6
        )
        nbytes = buf.getbuffer().nbytes
        buf.seek(0)
        return np.array(Image.open(buf).convert('RGB')), nbytes

    def _jpeg2000_roundtrip(self, img_u8: np.ndarray) -> tuple[np.ndarray, int]:
        psnr_list = [20.0, 24.0, 28.0, 32.0, 36.0, 40.0]
        cr_list = [500, 200, 100, 50, 25, 12]
        idx = min(max(self.q_level, 1), 6) - 1
        target_psnr = psnr_list[idx]
        cr = cr_list[idx]

        if _ic is not None:
            try:
                encoded = _ic.jpeg2k_encode(
                    img_u8,
                    psnr=target_psnr,
                    irreversible=True,
                    mct=1,
                )
                nbytes = len(encoded)
                return _ic.jpeg2k_decode(encoded), nbytes
            except Exception:
                pass

        tmp = np.ascontiguousarray(img_u8)
        with BytesIO() as buf:
            for writer_kwargs in [
                {'psnr': target_psnr},
                {'cratio': cr},
                {'rate': max(0.05, 8.0 / cr)},
                {},
            ]:
                try:
                    iio.imwrite(buf, tmp, extension='.jp2', **writer_kwargs)
                    nbytes = buf.getbuffer().nbytes
                    buf.seek(0)
                    return iio.imread(buf), nbytes
                except Exception:
                    buf.seek(0)
                    buf.truncate()
        raise RuntimeError("JPEG2000 encoding failed with all backends")

    def _jpegxl_roundtrip(self, img_u8: np.ndarray) -> tuple[np.ndarray, int]:
        dist_map = [4.0, 3.0, 2.0, 1.5, 1.0, 0.6]
        dist = (float(self._jpegxl_distance) if self._jpegxl_distance is not None
                else dist_map[min(max(self.q_level, 1), 6) - 1])

        if _ic is not None:
            try:
                encoded = _ic.jpegxl_encode(img_u8, distance=dist, effort=7)
                nbytes = len(encoded)
                return _ic.jpegxl_decode(encoded), nbytes
            except Exception:
                pass

        try:
            import subprocess as sp
            import tempfile as tf
            with tf.TemporaryDirectory() as td:
                inp = os.path.join(td, 'in.png')
                jxl = os.path.join(td, 'out.jxl')
                outp = os.path.join(td, 'out.png')
                iio.imwrite(inp, img_u8, extension='.png')
                sp.run(
                    ['cjxl', inp, jxl, f'--distance={dist}', '--effort=7'],
                    check=True,
                    stdout=sp.DEVNULL,
                    stderr=sp.DEVNULL,
                )
                nbytes = os.path.getsize(jxl)
                sp.run(
                    ['djxl', jxl, outp],
                    check=True,
                    stdout=sp.DEVNULL,
                    stderr=sp.DEVNULL,
                )
                return iio.imread(outp), nbytes
        except Exception:
            pass

        with BytesIO() as buf:
            for kwargs in [{'distance': dist}, {}]:
                try:
                    iio.imwrite(buf, img_u8, extension='.jxl', **kwargs)
                    nbytes = buf.getbuffer().nbytes
                    buf.seek(0)
                    return iio.imread(buf), nbytes
                except Exception:
                    buf.seek(0)
                    buf.truncate()
        raise RuntimeError("JPEG XL encoding failed with all backends")


def load_compressai_model(model_name: str, quality: int, device: torch.device):
    """Load a pretrained CompressAI model."""
    from compressai.zoo import models as compressai_models

    model_class = compressai_models.get(model_name, None)
    if not model_class:
        raise ValueError(
            f"Model {model_name} not found in compressai.zoo.models"
        )

    torch.cuda.empty_cache()
    model = model_class(quality=quality, pretrained=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_tcm_model(p: int, device: torch.device, base_dir: str):
    """Load and configure a TCM model."""

    base_dir_p = Path(base_dir).expanduser()
    tcm_path = (base_dir_p / "LIC_TCM").resolve()

    try:
        from .tcm_setup import ensure_tcm_assets

        ensure_tcm_assets(tcm_dir=tcm_path)
    except Exception as e:
        raise RuntimeError(
            "TCM assets are missing and auto-setup failed. "
            "Try running `python setup.py` from the repo root, "
            "or ensure git + gdown are available. "
            f"Details: {e}"
        ) from e
    tcm_path_s = str(tcm_path)

    # Clean up any cached models modules that might conflict
    try:
        modules_to_remove = []
        for k in list(sys.modules.keys()):
            if k == "models" or k.startswith("models."):
                # Check if this module is from a different path
                mod = sys.modules[k]
                mod_file = getattr(mod, "__file__", None)
                if mod_file and tcm_path_s not in mod_file:
                    modules_to_remove.append(k)
        for k in modules_to_remove:
            del sys.modules[k]
    except Exception:
        pass

    # Ensure TCM path is at the beginning
    if tcm_path_s in sys.path:
        sys.path.remove(tcm_path_s)
    sys.path.insert(0, tcm_path_s)

    checkpoint_map = {
        128: str(tcm_path / "mse_lambda_0.05.pth.tar"),
        64: str(tcm_path / "mse_lambda_0.0025.pth.tar"),
    }
    if p not in checkpoint_map:
        raise ValueError(
            f"Unsupported p value: {p}. Supported: {list(checkpoint_map.keys())}"
        )

    checkpoint = torch.load(checkpoint_map[p], map_location=device)
    state_dict = {
        k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
    }

    from models.tcm import TCM
    model = TCM(
        config=[2, 2, 2, 2, 2, 2],
        head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0.0,
        N=p,
        M=320,
    ).to(device).eval()

    model.load_state_dict(state_dict)
    model.update()
    return model


def load_ftic_model(quality: int, device: torch.device, base_dir: str):
    """Load FTIC model (ICLR2024)."""
    base_dir_p = Path(base_dir).expanduser()
    ftic_path = (base_dir_p / "ICLR2024-FTIC").resolve()

    checkpoint_map = {
        1: "ckpt_mse_0018.pth",
        2: "ckpt_mse_0035.pth",
        3: "ckpt_mse_0067.pth",
        4: "ckpt_mse_0130.pth",
        5: "ckpt_mse_0250.pth",
        6: "ckpt_mse_0483.pth",
    }

    if quality not in checkpoint_map:
        raise ValueError(f"Quality {quality} not in {list(checkpoint_map.keys())}")

    ckpt_path = ftic_path / "checkpoints" / checkpoint_map[quality]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ftic_path_s = str(ftic_path)

    # Clean up any cached models modules that might conflict
    try:
        modules_to_remove = []
        for k in list(sys.modules.keys()):
            if k == "models" or k.startswith("models."):
                # Check if this module is from a different path
                mod = sys.modules[k]
                mod_file = getattr(mod, "__file__", None)
                if mod_file and ftic_path_s not in mod_file:
                    modules_to_remove.append(k)
        for k in modules_to_remove:
            del sys.modules[k]
    except Exception:
        pass

    # Ensure FTIC path is at the beginning
    if ftic_path_s in sys.path:
        sys.path.remove(ftic_path_s)
    sys.path.insert(0, ftic_path_s)

    # Now import
    from models.flic import FrequencyAwareTransFormer

    model = FrequencyAwareTransFormer().to(device).eval()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.update()

    for param in model.parameters():
        param.requires_grad = False

    return model


def load_model(
    model_name: str,
    quality: int,
    device: torch.device,
    p: int = 128,
    base_dir: str | None = None,
    classical_overrides: dict | None = None,
):
    """
    Universal model loader for NIC and traditional codecs.

    Args:
        model_name: 'ftic', 'tcm', CompressAI model name, or codec ('jpeg', 'webp', 'jpegxl', 'jpeg2000')
        quality: Quality level (1-6 for most models)
        device: torch device
        p: TCM-specific parameter (64 or 128)
        base_dir: Base directory for FTIC/TCM checkpoints
        classical_overrides: Optional dict for classical codecs, e.g. {'jpeg': 43, 'webp': 70, 'jpegxl': 2.5}
            to use specific quality/distance instead of the mapped q_level.
    """
    name = model_name.lower()

    if name in ImageCodecModel.SUPPORTED_CODECS:
        kwargs = {}
        if classical_overrides and name in classical_overrides:
            v = classical_overrides[name]
            if name == 'jpeg':
                kwargs['jpeg_quality'] = int(v)
            elif name == 'webp':
                kwargs['webp_quality'] = int(v)
            elif name == 'jpegxl':
                kwargs['jpegxl_distance'] = float(v)
        return ImageCodecModel(name, quality, **kwargs)
    elif name == 'tcm':
        if base_dir is None:
            base_dir = str(Path(__file__).resolve().parents[1] / "third_party")
        return load_tcm_model(p, device, base_dir)
    elif name == 'ftic':
        if base_dir is None:
            base_dir = str(Path(__file__).resolve().parents[1] / "third_party")
        return load_ftic_model(quality, device, base_dir)
    else:
        return load_compressai_model(model_name, quality, device)


def get_available_models(include_codecs: bool = True) -> list:
    """Return list of available model names."""
    try:
        from compressai.zoo import models as compressai_models
        zoo_models = [
            'cheng2020-anchor', 'cheng2020-attn',
            'bmshj2018-factorized', 'bmshj2018-hyperprior',
            'mbt2018-mean', 'mbt2018',
        ]
        available = [m for m in zoo_models if compressai_models.get(m, None)]
    except ImportError:
        available = []

    models = ['ftic', 'tcm'] + available
    if include_codecs:
        # jpeg2000 is often problematic / flaky across environments
        models += list(ImageCodecModel.SUPPORTED_CODECS - {'jpeg2000'})
    return models