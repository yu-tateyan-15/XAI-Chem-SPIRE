"""Io for notebook-based atom-contribution experiments."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
import json
from pathlib import Path
from typing import Any
from typing import BinaryIO
import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage


MAX_SAFE_FLOAT_INTEGER = (1 << 53) - 1


def exact_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    """Return an integer without silently truncating fractional values."""

    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be boolean")
    if isinstance(value, Real) and not isinstance(value, Integral):
        numeric = float(value)
        if abs(numeric) > MAX_SAFE_FLOAT_INTEGER:
            raise ValueError(
                f"{name} exceeds exact floating-point integer range; "
                "supply an integer or decimal string"
            )
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{name} must be an integer")
    integer = int(number)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer


def exact_integers(
    values: Iterable[object],
    *,
    name: str,
    minimum: int | None = None,
) -> tuple[int, ...]:
    """Validate an integer sequence while retaining the failing position."""

    return tuple(
        exact_integer(value, name=f"{name}[{position}]", minimum=minimum)
        for position, value in enumerate(values)
    )


def write_json(
    path: Path | str,
    payload: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
) -> Path:
    """Write stable, human-readable JSON with a trailing newline."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


_PYG_SAFE_GLOBALS = (Data, DataEdgeAttr, DataTensorAttr, GlobalStorage)


def load_pyg_data(
    source: Path | str | BinaryIO,
    *,
    map_location: str = "cpu",
) -> Data:
    """Load one exact PyG ``Data`` object without unrestricted pickle."""

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        raise RuntimeError(
            "PyG graph loading requires a PyTorch version with "
            "torch.serialization.safe_globals"
        )
    with safe_globals(list(_PYG_SAFE_GLOBALS)):
        value = torch.load(
            source,
            map_location=map_location,
            weights_only=True,
        )
    if type(value) is not Data:
        raise TypeError("Graph file must contain exactly one PyG Data object")
    return value
