"""Model loading and transcription — pure ML, no UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from .audio import TARGET_SR


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    n_chunks: int


def load_model(
    model_id: str, device: str, dtype: torch.dtype
) -> tuple[Any, CohereAsrForConditionalGeneration]:
    """Load processor + model and put the model in eval mode."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
    )
    model.eval()
    return processor, model


def transcribe(
    processor: Any,
    model: CohereAsrForConditionalGeneration,
    audio: np.ndarray,
    *,
    language: str,
    punctuation: bool = True,
    max_new_tokens: int = 440,
    temperature: Optional[float] = None,
    sampling_rate: int = TARGET_SR,
) -> TranscriptionResult:
    """Run a single end-to-end transcription pass.

    The processor automatically chunks audio longer than its internal
    `max_audio_clip_s` (~35s) and emits an `audio_chunk_index` tensor; we
    capture it before moving inputs to device so `processor.decode` can
    reassemble per-chunk transcripts in order.
    """
    inputs = processor(
        audio=audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        language=language,
        punctuation=punctuation,
    )
    audio_chunk_index = inputs.get("audio_chunk_index")
    n_chunks = (
        int(audio_chunk_index.shape[0])
        if audio_chunk_index is not None and hasattr(audio_chunk_index, "shape")
        else 1
    )
    inputs = inputs.to(model.device, dtype=model.dtype)

    gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    decoded = processor.decode(
        outputs,
        skip_special_tokens=True,
        audio_chunk_index=audio_chunk_index,
        language=language,
    )
    text = decoded[0] if isinstance(decoded, list) else decoded
    return TranscriptionResult(text=text, n_chunks=n_chunks)
