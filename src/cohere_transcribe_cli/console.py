"""Rich consoles + small CLI presentation helpers."""

from __future__ import annotations

import contextlib
from typing import ContextManager

import typer
from rich.console import Console

stdout = Console()
stderr = Console(stderr=True)


def fail(msg: str, code: int = 1) -> None:
    """Print a red error to stderr and exit."""
    stderr.print(f"[bold red]error[/] {msg}")
    raise typer.Exit(code)


def human_size(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n / 1024:.1f} TB"


def status(msg: str, *, quiet: bool, spinner: str = "dots") -> ContextManager:
    """Spinner context manager that becomes a nullcontext when quiet."""
    if quiet:
        return contextlib.nullcontext()
    return stderr.status(msg, spinner=spinner)
