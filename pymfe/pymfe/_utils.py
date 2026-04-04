"""Keeps generic utility functions."""
import typing as t
import inspect

import numpy as np
import sklearn.preprocessing

from pymfe.pymfe import _backend


def calc_cls_inds(
    y: np.ndarray, classes: t.Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the ``cls_inds`` variable.

    The ``cls_inds`` variable is a boolean array which marks with
    True value whether the instance belongs to each class. Each
    distinct class is represented by a row, and each instance is
    represented by a column.
    """
    y = _backend.asarray(y)

    if classes is None:
        classes = _backend.unique(y)

    classes = _backend.asarray(classes)
    xp = _backend.get_array_module(y, classes)

    cls_inds = xp.array([xp.equal(y, cur_cls) for cur_cls in classes], dtype=bool)

    return cls_inds


def get_one_hot_encoder(**kwargs: t.Any) -> sklearn.preprocessing.OneHotEncoder:
    ohe_kwarg_names = inspect.getfullargspec(sklearn.preprocessing.OneHotEncoder).kwonlyargs
    ohe_kwargs = {"drop": "first"}
    ohe_kwargs["sparse_output" if "sparse_output" in ohe_kwarg_names else "sparse"] = False
    ohe_kwargs.update(kwargs)
    return sklearn.preprocessing.OneHotEncoder(**ohe_kwargs)

