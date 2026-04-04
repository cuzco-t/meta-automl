"""Optional GPU backend helpers for CuPy/cuDF integrations."""

from __future__ import annotations

import typing as t

import numpy as np

try:  # pragma: no cover - optional dependency
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import cupyx.scipy.spatial.distance as cupy_distance
except Exception:  # pragma: no cover - optional dependency
    cupy_distance = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import cudf
except Exception:  # pragma: no cover - optional dependency
    cudf = None  # type: ignore[assignment]


def has_gpu() -> bool:
    """Return True when CuPy is importable."""

    return cp is not None


def is_gpu_array(value: t.Any) -> bool:
    """Return True when value is a CuPy array."""

    return cp is not None and isinstance(value, cp.ndarray)


def get_array_module(*values: t.Any):
    """Return NumPy or CuPy depending on the provided values."""

    if cp is not None and any(is_gpu_array(value) for value in values):
        return cp

    return np


def asarray(value: t.Any, dtype: t.Optional[t.Any] = None):
    """Convert a value to a backend array when possible."""

    if value is None:
        return None

    if is_gpu_array(value):
        return cp.asarray(value, dtype=dtype)

    if cudf is not None and isinstance(value, (cudf.Series, cudf.DataFrame)):
        if hasattr(value, "to_cupy") and cp is not None:
            return cp.asarray(value.to_cupy(), dtype=dtype)

        return np.asarray(value.to_pandas(), dtype=dtype)

    if cp is not None and not isinstance(value, np.ndarray):
        try:
            return cp.asarray(value, dtype=dtype)
        except Exception:
            pass

    return np.asarray(value, dtype=dtype)


def to_cpu(value: t.Any):
    """Convert backend arrays back to NumPy arrays when needed."""

    if cp is not None and isinstance(value, cp.ndarray):
        return cp.asnumpy(value)

    if cudf is not None and isinstance(value, (cudf.Series, cudf.DataFrame)):
        return value.to_pandas().to_numpy()

    return value


def unique(values: t.Any, return_counts: bool = False):
    """Backend-aware unique helper."""

    xp = get_array_module(values)
    values = asarray(values)

    if xp is np:
        return np.unique(values, return_counts=return_counts)

    return xp.unique(values, return_counts=return_counts)


def bincount(values: t.Any):
    """Backend-aware bincount helper."""

    xp = get_array_module(values)
    values = asarray(values)

    if xp is np:
        return np.bincount(values)

    return xp.bincount(values)


def cov(values: t.Any, rowvar: bool = False, ddof: int = 1):
    """Backend-aware covariance helper."""

    xp = get_array_module(values)
    values = asarray(values, dtype=float)

    if xp is np:
        return np.cov(values, rowvar=rowvar, ddof=ddof)

    return xp.cov(values, rowvar=rowvar, ddof=ddof)


def corrcoef(values: t.Any, rowvar: bool = False):
    """Backend-aware correlation helper."""

    xp = get_array_module(values)
    values = asarray(values, dtype=float)

    if xp is np:
        return np.corrcoef(values, rowvar=rowvar)

    return xp.corrcoef(values, rowvar=rowvar)


def cdist(
    x_values: t.Any,
    y_values: t.Any,
    metric: str = "euclidean",
    p: t.Union[int, float] = 2,
):
    """Backend-aware pairwise distance helper."""

    xp = get_array_module(x_values, y_values)

    if xp is np or cupy_distance is None:
        from scipy.spatial.distance import cdist as scipy_cdist

        return scipy_cdist(
            np.asarray(to_cpu(x_values)),
            np.asarray(to_cpu(y_values)),
            metric=metric,
            p=p,
        )

    return cupy_distance.cdist(asarray(x_values), asarray(y_values), metric=metric, p=p)
