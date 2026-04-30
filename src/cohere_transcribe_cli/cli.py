"""Colored CLI that runs Cohere Transcribe locally via Hugging Face Transformers."""

from __future__ import annotations

import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import av
import numpy as np
import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_LANGUAGE = "en"
TARGET_SR = 16000
SUPPORTED_LANGUAGES = (
    "en", "de", "fr", "it", "es", "pt", "el",
    "nl", "pl", "vi", "zh", "ar", "ja", "ko",
)


class Device(str, Enum):
    auto = "auto"
    cuda = "cuda"
    mps = "mps"
    cpu = "cpu"


class Dtype(str, Enum):
    auto = "auto"
    bf16 = "bf16"
    fp16 = "fp16"
    fp32 = "fp32"

stdout = Console()
stderr = Console(stderr=True)

app = typer.Typer(
    name="cohere",
    add_completion=False,
    rich_markup_mode="rich",
    help="Transcribe audio/video locally with [bold cyan]Cohere Transcribe[/].",
)


def _fail(msg: str, code: int = 1) -> None:
    stderr.print(f"[bold red]error[/] {msg}")
    raise typer.Exit(code)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _decode_audio(file: Path) -> np.ndarray:
    """Decode any media file's first audio stream into a 16 kHz mono float32 array."""
    chunks: list[np.ndarray] = []
    with av.open(str(file)) as container:
        in_stream = next(
            (s for s in container.streams if s.type == "audio"), None
        )
        if in_stream is None:
            raise ValueError(f"no audio stream found in {file.name}")

        resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)

        def _drain(frames):
            for resampled in frames:
                arr = resampled.to_ndarray()
                chunks.append(arr.reshape(-1).astype(np.float32, copy=False))

        for frame in container.decode(in_stream):
            _drain(resampler.resample(frame))
        _drain(resampler.resample(None))

    if not chunks:
        raise ValueError(f"no audio decoded from {file.name}")
    return np.concatenate(chunks)


def _resolve_device_dtype(
    device_opt: Device, dtype_opt: Dtype
) -> tuple[str, torch.dtype]:
    if device_opt is Device.auto:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_opt.value

    if dtype_opt is Dtype.auto:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = {
            Dtype.bf16: torch.bfloat16,
            Dtype.fp16: torch.float16,
            Dtype.fp32: torch.float32,
        }[dtype_opt]
    return device, dtype


@app.command()
def main(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, readable=True,
            help="Audio or video file to transcribe.",
        ),
    ],
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="ISO-639-1 language code."),
    ] = DEFAULT_LANGUAGE,
    model_id: Annotated[
        str,
        typer.Option("--model", "-m", help="Hugging Face model id."),
    ] = DEFAULT_MODEL,
    temperature: Annotated[
        Optional[float],
        typer.Option("--temperature", "-t", min=0.0, max=2.0,
                     help="Sampling temperature (0 = greedy)."),
    ] = None,
    max_new_tokens: Annotated[
        int,
        typer.Option("--max-new-tokens", min=1, help="Generation cap."),
    ] = 440,
    punctuation: Annotated[
        bool,
        typer.Option("--punctuation/--no-punctuation",
                     help="Emit cased & punctuated text."),
    ] = True,
    device: Annotated[
        Device,
        typer.Option("--device", "-d", help="Inference device."),
    ] = Device.auto,
    dtype: Annotated[
        Dtype,
        typer.Option("--dtype", help="Model dtype (auto = bf16 on CUDA, else fp32)."),
    ] = Dtype.auto,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Write transcript to a file (else stdout)."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress decorative output."),
    ] = False,
) -> None:
    """Transcribe a single audio or video file locally."""
    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        _fail(f"unsupported language [yellow]{language}[/]. supported: {supported}")

    try:
        resolved_device, resolved_dtype = _resolve_device_dtype(device, dtype)
    except KeyError as exc:
        _fail(f"unknown dtype: {exc}")

    if not quiet:
        size = _human_size(file.stat().st_size)
        stderr.print(
            f"[dim]→[/] [bold]{file.name}[/] "
            f"[dim]({size}, lang={language}, device={resolved_device}, "
            f"dtype={str(resolved_dtype).removeprefix('torch.')})[/]"
        )

    # Decode audio
    decode_started = time.monotonic()
    try:
        with (stderr.status("[cyan]decoding audio…[/]", spinner="dots")
              if not quiet else _nullctx()):
            audio = _decode_audio(file)
    except (ValueError, av.error.InvalidDataError) as exc:
        _fail(str(exc))
    decode_elapsed = time.monotonic() - decode_started
    duration_s = len(audio) / TARGET_SR
    if not quiet:
        stderr.print(
            f"[dim]  decoded {duration_s:.1f}s of audio in {decode_elapsed:.1f}s[/]"
        )

    # Load model + processor (cached after first run by HF)
    load_started = time.monotonic()
    try:
        with (stderr.status(
                f"[cyan]loading {model_id}…[/]", spinner="dots")
              if not quiet else _nullctx()):
            processor = AutoProcessor.from_pretrained(model_id)
            model = CohereAsrForConditionalGeneration.from_pretrained(
                model_id, device_map=resolved_device, dtype=resolved_dtype,
            )
            model.eval()
    except Exception as exc:  # HF surfaces many error types
        _fail(f"failed to load model: {exc}")
    load_elapsed = time.monotonic() - load_started
    if not quiet:
        stderr.print(f"[dim]  model ready in {load_elapsed:.1f}s[/]")

    # Inference
    gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    infer_started = time.monotonic()
    try:
        with (stderr.status("[cyan]transcribing…[/]", spinner="dots")
              if not quiet else _nullctx()):
            inputs = processor(
                audio, sampling_rate=TARGET_SR, return_tensors="pt",
                language=language, punctuation=punctuation,
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)
            text = processor.decode(outputs, skip_special_tokens=True)
            if isinstance(text, list):
                text = text[0] if text else ""
    except Exception as exc:
        _fail(f"inference failed: {exc}", code=2)
    infer_elapsed = time.monotonic() - infer_started

    rtf = infer_elapsed / duration_s if duration_s > 0 else float("inf")
    if output:
        output.write_text(text)
        if not quiet:
            stderr.print(
                f"[bold green]✓[/] wrote [cyan]{output}[/] "
                f"[dim]({len(text)} chars, infer={infer_elapsed:.1f}s, rtf={rtf:.2f}×)[/]"
            )
    else:
        if quiet:
            stdout.print(text, highlight=False)
        else:
            stderr.print(
                f"[bold green]✓[/] done "
                f"[dim]({len(text)} chars, infer={infer_elapsed:.1f}s, rtf={rtf:.2f}×)[/]"
            )
            stdout.print(Panel(Text(text), title="transcript", border_style="green"))


def _nullctx():
    import contextlib
    return contextlib.nullcontext()


if __name__ == "__main__":
    app()
