"""Rich progress display for benchmark runs."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


@contextmanager
def benchmark_progress(
    console: Console | None = None,
) -> Generator[Progress, None, None]:
    """Context manager that yields a Rich Progress instance for benchmark tracking."""
    console = console or Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[dim]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        yield progress


def make_on_step(
    progress: Progress,
    warmup_task_id: TaskID,
    bench_task_id: TaskID,
) -> Callable[[str, int, int], None]:
    """Create an on_step callback that updates progress bar tasks."""

    def on_step(phase: str, current: int, total: int) -> None:
        if phase == "warmup":
            progress.update(warmup_task_id, completed=current, total=total)
        else:
            progress.update(bench_task_id, completed=current, total=total)

    return on_step
