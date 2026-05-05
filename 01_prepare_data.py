import os
import csv
import yaml
import argparse
import pandas as pd
from pathlib import Path


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def parse_words_csv(csv_path: Path) -> list:
    """
    Parse a _words.csv file and return a list of dicts with keys:
    word, t_start, t_end.

    The file uses semicolons as separators and has no header.
    Empty labels (silence) are kept so we can filter them downstream.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for line in reader:
            if len(line) < 3:
                continue
            word    = line[0].strip().strip('"')   # remove surrounding quotes
            t_start = float(line[1])
            t_end   = float(line[2])
            rows.append({"word": word, "t_start": t_start, "t_end": t_end})
    return rows


def resolve_dev_speakers(
    all_speaker_dirs: list,
    dev_speakers: list,
    dev_n_speakers: int,
) -> list:
    """
    Return the subset of speaker directories to use in dev mode.

    Parameters
    ----------
    all_speaker_dirs : sorted list of Path objects (all speaker folders)
    dev_speakers     : explicit list of initials from params, e.g. ["ab", "mv"]
    dev_n_speakers   : fallback count when dev_speakers is empty

    Returns
    -------
    List of Path objects for the selected speakers.
    """
    all_names = [d.name for d in all_speaker_dirs]

    if dev_speakers:
        # User provided an explicit list — validate each entry
        selected = []
        for initials in dev_speakers:
            matches = [d for d in all_speaker_dirs if d.name == initials]
            if not matches:
                raise ValueError(
                    f"dev_speakers entry '{initials}' not found in corpus.\n"
                    f"Available speakers: {all_names}"
                )
            selected.append(matches[0])
        return selected
    else:
        # Auto-select first N speakers alphabetically
        n = min(dev_n_speakers, len(all_speaker_dirs))
        if n < 2:
            print(
                "WARNING: dev_n_speakers < 2. You need at least 2 speakers "
                "to compute inter-speaker distances."
            )
        return all_speaker_dirs[:n]


def main():
    parser = argparse.ArgumentParser(description="Build corpus manifest")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params      = load_params(args.params)
    corpus_root = Path(params["corpus_root"])
    skip_labels = set(params["skip_labels"])
    min_dur     = params["min_word_duration"]

    # Dev mode configuration
    dev_mode      = params.get("dev_mode", False)
    dev_speakers  = params.get("dev_speakers", [])
    dev_n_speakers = params.get("dev_n_speakers", 2)

    if not corpus_root.exists():
        raise FileNotFoundError(
            f"corpus_root does not exist: {corpus_root}\n"
            "Please set the correct path in params.yaml"
        )

    # Collect and sort all speaker directories
    all_speaker_dirs = sorted([d for d in corpus_root.iterdir() if d.is_dir()])
    print(f"Found {len(all_speaker_dirs)} speaker directories under {corpus_root}")

    # Apply dev mode filtering
    if dev_mode:
        speaker_dirs = resolve_dev_speakers(
            all_speaker_dirs, dev_speakers, dev_n_speakers
        )
        print(
            f"\n*** DEV MODE ACTIVE ***\n"
            f"Processing {len(speaker_dirs)} of {len(all_speaker_dirs)} speakers: "
            f"{[d.name for d in speaker_dirs]}\n"
            f"To run on the full corpus, set dev_mode: false in params.yaml\n"
        )
    else:
        speaker_dirs = all_speaker_dirs
        print("Full corpus mode: processing all speakers.")

    # Walk selected speaker directories
    records = []

    for spk_dir in speaker_dirs:
        speaker = spk_dir.name   # e.g. "ab"

        # Find all _words.csv files in this speaker's folder
        csv_files = sorted(spk_dir.glob("*_words.csv"))
        if not csv_files:
            print(f"  [{speaker}] No _words.csv found — skipping")
            continue

        for csv_path in csv_files:
            # Derive the companion WAV path by removing "_words" suffix
            # e.g. ab_rus_list1_FRcorp1_words.csv -> ab_rus_list1_FRcorp1.wav
            wav_name     = csv_path.stem.replace("_words", "") + ".wav"
            wav_path     = csv_path.parent / wav_name
            recording_id = csv_path.stem.replace("_words", "")

            if not wav_path.exists():
                print(f"  [{speaker}] WAV not found for {csv_path.name} — skipping")
                continue

            word_rows = parse_words_csv(csv_path)

            for row in word_rows:
                word     = row["word"]
                t_start  = row["t_start"]
                t_end    = row["t_end"]
                duration = t_end - t_start

                # Filter out silence / very short segments
                if word in skip_labels or duration < min_dur:
                    continue

                records.append({
                    "speaker":      speaker,
                    "recording_id": recording_id,
                    "word":         word.lower(),   # normalise case
                    "t_start":      t_start,
                    "t_end":        t_end,
                    "duration":     round(duration, 4),
                    "wav_path":     str(wav_path.resolve()),
                })

    if not records:
        raise RuntimeError(
            "Manifest is empty. Check corpus_root in params.yaml and folder structure.\n"
            + ("(dev_mode is ON — make sure the selected speakers have data)" if dev_mode else "")
        )

    df = pd.DataFrame(records)

    # Summary statistics

    mode_label = f"[DEV MODE — {len(speaker_dirs)} speakers]" if dev_mode else "[FULL CORPUS]"
    print(f"\n=== Manifest summary {mode_label} ===")
    print(f"Total word segments : {len(df)}")
    print(f"Unique speakers     : {df['speaker'].nunique()}  {df['speaker'].unique().tolist()}")
    print(f"Unique words        : {df['word'].nunique()}")
    print(f"\nTop 20 most frequent words:")
    print(df["word"].value_counts().head(20).to_string())

    # Words shared across multiple speakers (needed for inter-speaker distances)
    word_speaker_counts = df.groupby("word")["speaker"].nunique()
    shared_words        = word_speaker_counts[word_speaker_counts >= 2].index
    print(f"\nWords spoken by >=2 speakers : {len(shared_words)}")
    df_shared = df[df["word"].isin(shared_words)]
    print(f"Segments kept for analysis   : {len(df_shared)} / {len(df)}")



    # -------------------------------------------------------------------------
    # Save manifest — also records whether this is a dev run
    # -------------------------------------------------------------------------
    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"\nManifest saved -> {out_path}")

    # Write a small metadata sidecar so downstream stages can log dev/full mode
    meta = {
        "dev_mode":       dev_mode,
        "n_speakers":     int(df["speaker"].nunique()),
        "n_segments":     len(df),
        "n_shared_words": len(shared_words),
    }
    import json
    with open(out_dir / "manifest_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
