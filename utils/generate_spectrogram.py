import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert audio FLAC files to spectrogram"
    )
    parser.add_argument("--src_dir", type=str, help="directory containing flac files")
    parser.add_argument("--dest_dir", type=str, help="directory to store spectrograms")
    args = parser.parse_args()
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    src_files = Path(args.src_dir).glob("*.flac")
    for f in tqdm(src_files):
        sig, sr = sf.read(f)
        melspec = librosa.feature.melspectrogram(
            y=sig, sr=16000, hop_length=64, win_length=128
        )
        filename = f.name
        filename = os.path.join(dest_dir, f"{filename.split('.')[0]}.npy")
        np.save(filename, melspec)
