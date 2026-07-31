"""Crash-safe file writes.

Saving an index overwrites five files in place. Interrupting that (Ctrl-C, a
full disk, a killed container) used to leave a half written file behind and
destroy an index that was perfectly good a second earlier - which matters
because ``graphrag index --append`` saves over the directory it just loaded.
Writing to a sibling temporary file and renaming it into place makes each file
either the old version or the new one, never a mixture.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Union


def _temporary(path: Path):
    return tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")


def atomic_write_text(
    path: Union[str, Path], text: str, encoding: str = "utf-8"
) -> Path:
    """Write ``text`` to ``path`` without ever leaving it partially written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = _temporary(path)
    try:
        with os.fdopen(handle, "w", encoding=encoding, newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        _discard(temporary)
        raise
    return path


def atomic_write_binary(
    path: Union[str, Path], writer: Callable[[Any], None]
) -> Path:
    """Same contract as :func:`atomic_write_text` for binary writers.

    ``writer`` is handed an open binary file object (``np.savez_compressed``
    and friends accept one).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = _temporary(path)
    try:
        with os.fdopen(handle, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        _discard(temporary)
        raise
    return path


def _discard(temporary: str) -> None:
    try:
        os.unlink(temporary)
    except OSError:  # pragma: no cover - best effort cleanup
        pass
