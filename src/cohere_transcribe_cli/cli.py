"""Typer entrypoint that orchestrates the local Cohere Transcribe pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Optional

import av
import typer

from .audio import TARGET_SR, decode_audio
from .compute import Device, Dtype, dtype_name, resolve
from .console import fail, human_size, status, stderr, stdout
from .transcribe import load_model, transcribe

DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = (
    "en",
    "de",
    "fr",
    "it",
    "es",
    "pt",
    "el",
    "nl",
    "pl",
    "vi",
    "zh",
    "ar",
    "ja",
    "ko",
)

app = typer.Typer(
    name="cohere",
    add_completion=False,
    rich_markup_mode="rich",
    help="Transcribe audio/video locally with [bold cyan]Cohere Transcribe[/].",
)


@app.command()
def main(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
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
        typer.Option(
            "--temperature",
            "-t",
            min=0.0,
            max=2.0,
            help="Sampling temperature (0 = greedy).",
        ),
    ] = None,
    max_new_tokens: Annotated[
        int,
        typer.Option("--max-new-tokens", min=1, help="Per-chunk generation cap."),
    ] = 440,
    punctuation: Annotated[
        bool,
        typer.Option(
            "--punctuation/--no-punctuation", help="Emit cased & punctuated text."
        ),
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
        typer.Option(
            "--output", "-o", help="Write transcript to a file (else stdout)."
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress decorative output."),
    ] = False,
) -> None:
    """Transcribe a single audio or video file locally."""
    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        fail(f"unsupported language [yellow]{language}[/]. supported: {supported}")

    resolved_device, resolved_dtype = resolve(device, dtype)

    if not quiet:
        stderr.print(
            f"[dim]→[/] [bold]{file.name}[/] "
            f"[dim]({human_size(file.stat().st_size)}, lang={language}, "
            f"device={resolved_device}, dtype={dtype_name(resolved_dtype)})[/]"
        )

    # Decode audio
    t0 = time.monotonic()
    try:
        with status("[cyan]decoding audio…[/]", quiet=quiet):
            audio = decode_audio(file)
    except (ValueError, av.error.InvalidDataError) as exc:
        fail(str(exc))
    duration_s = len(audio) / TARGET_SR
    if not quiet:
        stderr.print(
            f"[dim]  decoded {duration_s:.1f}s of audio in {time.monotonic() - t0:.1f}s[/]"
        )

    # Load model + processor (cached after first run by HF)
    t0 = time.monotonic()
    try:
        with status(f"[cyan]loading {model_id}…[/]", quiet=quiet):
            processor, model = load_model(model_id, resolved_device, resolved_dtype)
    except Exception as exc:
        fail(f"failed to load model: {exc}")
    if not quiet:
        stderr.print(f"[dim]  model ready in {time.monotonic() - t0:.1f}s[/]")

    # Inference
    t0 = time.monotonic()
    try:
        with status("[cyan]transcribing…[/]", quiet=quiet):
            result = transcribe(
                processor,
                model,
                audio,
                language=language,
                punctuation=punctuation,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
    except Exception as exc:
        fail(f"inference failed: {exc}", code=2)
    infer_elapsed = time.monotonic() - t0

    if not quiet and result.n_chunks > 1:
        stderr.print(f"[dim]  reassembled {result.n_chunks} chunks[/]")

    text = result.text
    rtf = infer_elapsed / duration_s if duration_s > 0 else float("inf")
    summary = f"[dim]({len(text)} chars, infer={infer_elapsed:.1f}s, rtf={rtf:.2f}×)[/]"

    if output:
        output.write_text(text)
        if not quiet:
            stderr.print(f"[bold green]✓[/] wrote [cyan]{output}[/] {summary}")
    else:
        if not quiet:
            stderr.print(f"[bold green]✓[/] done {summary}")
        stdout.print(text, highlight=False)


if __name__ == "__main__":
    app()
