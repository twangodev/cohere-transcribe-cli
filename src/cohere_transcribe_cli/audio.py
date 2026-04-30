"""PyAV-based decoder: any media file → mono float32 numpy array."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

TARGET_SR = 16000


def decode_audio(file: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode the first audio stream of any media file to mono float32 at target_sr."""
    chunks: list[np.ndarray] = []
    with av.open(str(file)) as container:
        in_stream = next((s for s in container.streams if s.type == "audio"), None)
        if in_stream is None:
            raise ValueError(f"no audio stream found in {file.name}")

        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)

        def drain(frames) -> None:
            for resampled in frames:
                arr = resampled.to_ndarray().reshape(-1)
                chunks.append(arr.astype(np.float32, copy=False))

        for frame in container.decode(in_stream):
            drain(resampler.resample(frame))
        drain(resampler.resample(None))

    if not chunks:
        raise ValueError(f"no audio decoded from {file.name}")
    return np.concatenate(chunks)
