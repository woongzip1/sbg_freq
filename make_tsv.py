# make_tsv.py: 디렉토리 폴더를 지정해주면, 
# 하위 모든 오디오를 탐색하고 정보를 저장
# 경로 \t 길이 \t rms
# 이때 경로는 sorted 된 순서로 저장
# python make_tsv.py --root /dataset --out train.tsv

#!/usr/bin/env python3
"""
make_tsv.py

Recursively scans a root folder for audio files and writes a TSV file
containing: absolute_path<TAB>duration_sec<TAB>rms.

Usage
-----
python make_tsv.py --root /dataset --out train.tsv
"""
import argparse
import os
from pathlib import Path
import soundfile as sf
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}

# -----------------------------------------------------------------------------#
def find_audio_files(root: Path):
    """Yield absolute paths to audio files under root (depth‑first)."""
    for dp, _, files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() in AUDIO_EXTS:
                yield Path(dp) / fn


def analyze_file(path: Path):
    """
    Return (str_path, duration_sec, rms) for a single file.
    On error, return None.
    """
    try:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        rms = float(np.sqrt(np.mean(data ** 2)))
        dur = float(len(data) / sr)
        return str(path), dur, rms
    except Exception:
        return None


# -----------------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory to scan")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Number of parallel processes (default: all cores)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_tsv = Path(args.out).expanduser().resolve()
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(find_audio_files(root))
    if not files:
        print("No audio found under", root)
        return

    print(f"Found {len(files)} audio files → processing with {args.workers} workers")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(analyze_file, p): p for p in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Analyzing"):
            res = fut.result()
            if res:
                results.append(res)

    # sort by path for determinism
    results.sort(key=lambda x: x[0])

    with out_tsv.open("w", encoding="utf-8") as f:
        for path, dur, rms in results:
            f.write(f"{path}\t{dur:.5f}\t{rms:.6f}\n")

    print(f"TSV saved to {out_tsv} (total {len(results)} lines)")


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()

# python make_tsv.py --root /home/woongzip/dataset_real/GT --out ./tsv_files/12k_path.tsv

# python make_tsv.py --root /home/woongzip/dataset_12/USAC44_GT --out ./tsv_files/val_gt.tsv;
# python make_tsv.py --root /home/woongzip/dataset_12/USAC44_20_core --out ./tsv_files/val_20k.tsv;
# python make_tsv.py --root /home/woongzip/Dataset/DAPS_gt_small --out ./tsv_files/val_speech_gt.tsv;
# python make_tsv.py --root /home/woongzip/Dataset/DAPS_12_core --out ./tsv_files/val_speech_12k.tsv;