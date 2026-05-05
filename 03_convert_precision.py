"""
Output files

data/features/embeddings_float64.npy   — upcast from float32
data/features/embeddings_float32.npy   — unchanged
data/features/embeddings_float16.npy   — downcast
data/features/embeddings_int8.npy      — quantised integers (int8)
data/features/scales_int8.npy          — per-vector scales for reconstruction
data/features/precision_sizes.json     — disk sizes in bytes for each format
"""

import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple



def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def quantise_int8(emb_f32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-vector symmetric linear quantisation to int8.

    Parameters
    emb_f32 : np.ndarray, shape (N, D), dtype float32

    Returns
    emb_int8  : np.ndarray, shape (N, D), dtype int8
                Quantised integers in [-127, 127]
    scales    : np.ndarray, shape (N,), dtype float32
                Per-vector scale factors for later reconstruction
    """
    # Compute per-vector maximum absolute value  →  shape (N, 1)
    abs_max = np.abs(emb_f32).max(axis=1, keepdims=True)   # (N, 1)

    # Avoid division by zero for all-zero vectors (shouldn't happen but safe)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)

    # Scale factors: maps [-abs_max, +abs_max] → [-127, +127]
    scales = (abs_max / 127.0).astype(np.float32)          # (N, 1)

    # Quantise: divide → round → clip to [-127, 127] → cast to int8
    q = np.round(emb_f32 / scales).clip(-127, 127).astype(np.int8)

    return q, scales.squeeze(1)  # scales shape: (N,)


def dequantise_int8(emb_int8: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Reconstruct float32 embeddings from int8 + scales.
    """
    return emb_int8.astype(np.float32) * scales[:, np.newaxis]


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def main():
    parser = argparse.ArgumentParser(description="Convert embeddings to multiple precisions")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)

    features_dir = Path("data/features")
    float32_path = features_dir / "embeddings_float32.npy"

    if not float32_path.exists():
        raise FileNotFoundError("Run stage 2 first: python src/02_extract_features.py")

    # Load float32 embeddings
    emb_f32 = np.load(float32_path)
    N, D = emb_f32.shape
    print(f"Loaded float32 embeddings: shape={emb_f32.shape}, dtype={emb_f32.dtype}")
    print(f"Memory footprint (float32): {emb_f32.nbytes / 1e6:.2f} MB")

    sizes = {}

    # 1. float64 — upcast for maximum reference precision

    emb_f64 = emb_f32.astype(np.float64)
    out_path = features_dir / "embeddings_float64.npy"
    np.save(out_path, emb_f64)
    sizes["float64"] = file_size_mb(out_path)
    print(f"\nfloat64 saved  ({sizes['float64']:.2f} MB) — 2× larger than float32")


    # 2. float32 — record size of existing file
    sizes["float32"] = file_size_mb(float32_path)
    print(f"float32 exists ({sizes['float32']:.2f} MB) — reference for ML pipelines")


    # 3. float16 — standard half-precision downcast
    # np.float16: 5 exponent bits, 10 mantissa bits, max value ~65504
    # Values outside [-65504, 65504] overflow to ±inf.
    # wav2vec2 outputs are typically in [-10, 10] so overflow is unlikely,
    # but precision loss (rounding to nearest representable float16) is significant.
    emb_f16 = emb_f32.astype(np.float16)

    # Check for overflows
    n_inf = np.isinf(emb_f16).sum()
    if n_inf > 0:
        print(f"  WARNING: {n_inf} values overflowed to ±inf in float16!")

    out_path = features_dir / "embeddings_float16.npy"
    np.save(out_path, emb_f16)
    sizes["float16"] = file_size_mb(out_path)
    print(f"float16 saved  ({sizes['float16']:.2f} MB) — 2× smaller than float32")

    # Quantisation error for float16
    err_f16 = np.abs(emb_f16.astype(np.float32) - emb_f32)
    print(f"  float16 max error: {err_f16.max():.6f}  mean error: {err_f16.mean():.6f}")

    # 4. int8 — our custom per-vector symmetric quantisation
    emb_int8, scales = quantise_int8(emb_f32)

    # Verify reconstruction quality
    emb_reconstructed = dequantise_int8(emb_int8, scales)
    err_int8 = np.abs(emb_reconstructed - emb_f32)
    print(f"\nint8 quantisation:")
    print(f"  max reconstruction error : {err_int8.max():.6f}")
    print(f"  mean reconstruction error: {err_int8.mean():.6f}")
    print(f"  scale statistics — min: {scales.min():.5f}  max: {scales.max():.5f}  mean: {scales.mean():.5f}")

    out_path  = features_dir / "embeddings_int8.npy"
    scale_path = features_dir / "scales_int8.npy"
    np.save(out_path,   emb_int8)
    np.save(scale_path, scales)
    sizes["int8"] = file_size_mb(out_path) + file_size_mb(scale_path)
    print(f"int8 saved     ({sizes['int8']:.2f} MB including scales)")

    out_json = features_dir / "precision_sizes.json"
    with open(out_json, "w") as f:
        json.dump(sizes, f, indent=2)

    print(f"\n=== Disk footprint summary ===")
    for fmt, mb in sizes.items():
        ratio = mb / sizes["float32"]
        print(f"  {fmt:8s}: {mb:6.2f} MB  ({ratio:.2f}× float32)")

    print(f"\nAll files saved under {features_dir}/")


if __name__ == "__main__":
    main()
