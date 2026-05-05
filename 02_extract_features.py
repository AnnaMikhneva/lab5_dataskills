import os
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Dict, Tuple



def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_audio(wav_path: str, target_sr: int = 16000):
    """
    Load audio and resample to 16 kHz (wav2vec2 requirement).
    Returns: (waveform_np float32, sample_rate)
    """
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)

    # Convert stereo to mono by averaging channels
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if necessary
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def extract_word_embedding(
    audio: np.ndarray,
    sr: int,
    t_start: float,
    t_end: float,
    model: Wav2Vec2Model,
    processor: Wav2Vec2Processor,
    pooling: str = "mean",
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract a single word-level embedding.

    Steps:
    1. Slice the raw waveform to [t_start, t_end]
    2. Run through wav2vec2 processor (normalisation)
    3. Run through wav2vec2 model
    4. Pool the frame-level hidden states → one vector of shape (D,)

    Note: wav2vec2-base produces hidden states of dimension D=768.
    """
    # 1. Slice audio to word boundaries
    i_start = int(t_start * sr)
    i_end   = int(t_end   * sr)
    segment = audio[i_start:i_end]


    if len(segment) < 10:
        return None

    # 2. Processor: normalises waveform and converts to tensor
    inputs = processor(
        segment,
        sampling_rate=sr,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs["input_values"].to(device)

    # 3. Forward pass — no gradient needed (inference only)
    with torch.no_grad():
        outputs = model(input_values)

    # hidden_states: (1, T, D)  where T = number of frames
    hidden_states = outputs.last_hidden_state  # shape: (1, T, D)

    # 4. Aggregate over time dimension
    if pooling == "mean":
        embedding = hidden_states.squeeze(0).mean(dim=0)          # (D,)
    elif pooling == "max":
        embedding = hidden_states.squeeze(0).max(dim=0).values    # (D,)
    elif pooling == "first_last":
        # Concatenate first and last frame → 2D vector
        h = hidden_states.squeeze(0)
        embedding = torch.cat([h[0], h[-1]], dim=0)               # (2D,)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    return embedding.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract wav2vec2 embeddings")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params     = load_params(args.params)
    model_name = params["model_name"]
    pooling    = params["pooling"]

    # Determine device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {model_name}  |  Pooling: {pooling}")

    # Load manifest
    manifest_path = Path("data/features/manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError("Run stage 1 first: python src/01_prepare_data.py")
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(df)} segments")

    # Load wav2vec2 model and processor from HuggingFace
    # The processor handles feature extraction (normalisation, padding).
    # The model is the neural network itself.
    print(f"Loading {model_name} from HuggingFace (downloads on first run)…")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()   # disable dropout for deterministic outputs

    # Cache loaded audio files to avoid re-reading the same WAV repeatedly
    audio_cache: Dict[str, Tuple] = {}

    embeddings = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        wav_path = row["wav_path"]

        # Load audio (with caching)
        if wav_path not in audio_cache:
            try:
                audio, sr = load_audio(wav_path)
                audio_cache[wav_path] = (audio, sr)
            except Exception as e:
                print(f"  Warning: could not load {wav_path}: {e}")
                continue

        audio, sr = audio_cache[wav_path]

        emb = extract_word_embedding(
            audio, sr,
            row["t_start"], row["t_end"],
            model, processor,
            pooling=pooling,
            device=device,
        )

        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(idx)

    print(f"\nSuccessfully extracted {len(embeddings)} embeddings")
    print(f"Skipped {len(df) - len(embeddings)} segments (too short / load error)")

    # Stack into matrix: shape (N, D)
    emb_matrix = np.stack(embeddings, axis=0)  # float32
    print(f"Embedding matrix shape: {emb_matrix.shape}  dtype: {emb_matrix.dtype}")

    # Save embeddings and the corresponding metadata rows
    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings_float32.npy", emb_matrix)
    df.iloc[valid_indices].reset_index(drop=True).to_csv(
        out_dir / "metadata.csv", index=False
    )

    print(f"\nSaved:")
    print(f"  {out_dir}/embeddings_float32.npy  ({emb_matrix.nbytes / 1e6:.1f} MB)")
    print(f"  {out_dir}/metadata.csv")


if __name__ == "__main__":
    main()
