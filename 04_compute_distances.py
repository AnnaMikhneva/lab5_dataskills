"""
For each precision format, computes:
  - The full pairwise cosine distance matrix
  - Intra-speaker distances  (same speaker, same word, different recording)
  - Inter-speaker distances  (different speakers, same word)
  - Their ratio
  - Time taken to compute the matrix
  - A comparison against the float64 reference (absolute deviation)


Outputs
data/distances/distances_<format>.npy     — full N×N distance matrix
data/distances/pairs_intra.csv            — intra-speaker pairs with distances
data/distances/pairs_inter.csv            — inter-speaker pairs with distances
data/distances/timing.json               — wall-clock time per format (seconds)
data/distances/summary_stats.csv         — mean intra/inter/ratio per format
"""

import json
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from typing import Tuple


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:

    # L2-normalise each row
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid /0
    emb_norm = emb / norms                      # (N, D)

    # Cosine similarity matrix
    sim = emb_norm @ emb_norm.T                 # (N, N)

    # Convert to distance and clip numerical noise
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)

    # Zero the diagonal (self-distance should be exactly 0)
    np.fill_diagonal(dist, 0.0)

    return dist


def load_embeddings(features_dir: Path, fmt: str) -> np.ndarray:
    """
    Load embeddings for a given format. For int8, load integers + scales
    and reconstruct float32 before computing distances.
    (Computing cosine distance on int8 directly would require special handling;
     dequantisation first is cleaner and equivalent for analysis purposes.)
    """
    if fmt == "int8":
        emb_int8 = np.load(features_dir / "embeddings_int8.npy")
        scales   = np.load(features_dir / "scales_int8.npy")
        # Reconstruct: cast to float32 then multiply by per-vector scale
        return emb_int8.astype(np.float32) * scales[:, np.newaxis]
    else:
        path = features_dir / f"embeddings_{fmt}.npy"
        return np.load(path)


def build_pairs(metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build intra-speaker and inter-speaker index pairs.

    Intra-speaker: same speaker, same word, different recording_id
    Inter-speaker: different speakers, same word

    Returns two DataFrames with columns [i, j, speaker_i, speaker_j, word].
    """
    intra_pairs = []
    inter_pairs = []

    # Group by word
    for word, grp in metadata.groupby("word"):
        indices   = grp.index.tolist()
        speakers  = grp["speaker"].tolist()
        rec_ids   = grp["recording_id"].tolist()

        for (pos_a, pos_b) in combinations(range(len(indices)), 2):
            i = indices[pos_a]
            j = indices[pos_b]
            spk_i = speakers[pos_a]
            spk_j = speakers[pos_b]
            rec_i = rec_ids[pos_a]
            rec_j = rec_ids[pos_b]

            if spk_i == spk_j:
                # Same speaker, same word, different recording → intra
                if rec_i != rec_j:
                    intra_pairs.append({"i": i, "j": j,
                                        "speaker": spk_i, "word": word})
            else:
                # Different speakers, same word → inter
                inter_pairs.append({"i": i, "j": j,
                                    "speaker_i": spk_i, "speaker_j": spk_j,
                                    "word": word})

    return pd.DataFrame(intra_pairs), pd.DataFrame(inter_pairs)


def main():
    parser = argparse.ArgumentParser(description="Compute distance matrices")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)

    features_dir  = Path("data/features")
    distances_dir = Path("data/distances")
    distances_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = pd.read_csv(features_dir / "metadata.csv")
    # Reset index to ensure it matches embedding row indices
    metadata = metadata.reset_index(drop=True)
    print(f"Metadata: {len(metadata)} segments")

    # Build pair lists (computed once, used for all precision levels)
    print("Building intra/inter speaker pairs…")
    intra_df, inter_df = build_pairs(metadata)
    print(f"  Intra-speaker pairs: {len(intra_df)}")
    print(f"  Inter-speaker pairs: {len(inter_df)}")

    if len(intra_df) == 0:
        print("\nWARNING: No intra-speaker pairs found.")
        print("This means no speaker repeated the same word across different recordings.")
        print("Check your corpus structure and manifest.")

    # Save pair lists (shared across all precision levels)
    intra_df.to_csv(distances_dir / "pairs_intra.csv", index=False)
    inter_df.to_csv(distances_dir / "pairs_inter.csv", index=False)


    formats = ["float64", "float32", "float16", "int8"]
    timing  = {}
    summary_rows = []

    # float64 distances are the reference; we will compare all others to it
    ref_dist_matrix = None

    for fmt in formats:
        print(f"\n--- {fmt} ---")

        emb = load_embeddings(features_dir, fmt)
        print(f"  Loaded embeddings: shape={emb.shape}  dtype={emb.dtype}")

        # Compute distance matrix and time it
        t0 = time.perf_counter()
        dist = cosine_distance_matrix(emb)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        timing[fmt] = elapsed
        print(f"  Distance matrix computed in {elapsed:.3f} s")

        # Save full matrix
        np.save(distances_dir / f"distances_{fmt}.npy", dist.astype(np.float64))

        # Store float64 matrix as reference
        if fmt == "float64":
            ref_dist_matrix = dist.copy()

        # Extract intra-speaker distances
        if len(intra_df) > 0:
            intra_distances = dist[intra_df["i"].values, intra_df["j"].values]
            mean_intra = float(np.mean(intra_distances))
            std_intra  = float(np.std(intra_distances))
        else:
            intra_distances = np.array([])
            mean_intra = np.nan
            std_intra  = np.nan

        # Extract inter-speaker distances
        if len(inter_df) > 0:
            inter_distances = dist[inter_df["i"].values, inter_df["j"].values]
            mean_inter = float(np.mean(inter_distances))
            std_inter  = float(np.std(inter_distances))
        else:
            inter_distances = np.array([])
            mean_inter = np.nan
            std_inter  = np.nan


        ratio = mean_intra / mean_inter if mean_inter > 0 else np.nan

        # Deviation from float64 reference
        if ref_dist_matrix is not None and fmt != "float64":
            deviation = float(np.mean(np.abs(dist - ref_dist_matrix)))
            max_dev   = float(np.abs(dist - ref_dist_matrix).max())
        else:
            deviation = 0.0
            max_dev   = 0.0

        print(f"  Mean intra-speaker distance : {mean_intra:.6f} ± {std_intra:.6f}")
        print(f"  Mean inter-speaker distance : {mean_inter:.6f} ± {std_inter:.6f}")
        print(f"  Intra/Inter ratio           : {ratio:.4f}")
        print(f"  Mean deviation from float64 : {deviation:.2e}")
        print(f"  Max  deviation from float64 : {max_dev:.2e}")

        # Save per-precision distance vectors (for plotting)
        np.save(distances_dir / f"intra_dists_{fmt}.npy", intra_distances)
        np.save(distances_dir / f"inter_dists_{fmt}.npy", inter_distances)

        summary_rows.append({
            "format":       fmt,
            "mean_intra":   mean_intra,
            "std_intra":    std_intra,
            "mean_inter":   mean_inter,
            "std_inter":    std_inter,
            "ratio":        ratio,
            "mean_dev_from_f64": deviation,
            "max_dev_from_f64":  max_dev,
            "time_s":       elapsed,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(distances_dir / "summary_stats.csv", index=False)

    with open(distances_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\n=== Summary ===")
    print(summary_df[["format", "mean_intra", "mean_inter", "ratio",
                       "mean_dev_from_f64", "time_s"]].to_string(index=False))

    print(f"\nAll distance files saved under {distances_dir}/")


if __name__ == "__main__":
    main()
