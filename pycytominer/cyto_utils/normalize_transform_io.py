"""
Save and load the fitted parameters of a normalize() transform.

Fitted parameters are stored as plain numeric arrays (in a numpy .npz
archive) plus a small JSON sidecar for scalar/string metadata, rather than
pickling the fitted scaler object itself. This lets a transform fit on one
set of samples (e.g. controls) be reapplied later to other subsets of data
without re-fitting.
"""

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from pycytominer.operations import RobustMAD, Spherize

avail_methods = ["standardize", "robustize", "mad_robustize", "spherize"]


def _npz_and_json_paths(file):
    """Derive the paired .npz (arrays) and .json (metadata) paths from a
    single user-provided path."""
    file = pathlib.Path(file)
    if file.suffix.lower() != ".npz":
        file = file.with_suffix(".npz")
    return file, file.with_suffix(".json")


def save_normalize_transform(scaler, method, features, output_file):
    """Save the fitted parameters of a normalize() scaler to disk.

    Parameters
    ----------
    scaler : fitted scaler object
        The fitted StandardScaler, RobustScaler, RobustMAD, or Spherize
        instance produced by the fitting step of `pycytominer.normalize`.
    method : str
        The normalization method used to fit `scaler`. One of "standardize",
        "robustize", "mad_robustize", "spherize".
    features : list of str
        The feature columns (in order) that the scaler was fit on.
    output_file : str or pathlib.Path
        Where to write the transform. Numeric parameters are written to this
        path with a ".npz" suffix; a companion ".json" file (same basename)
        stores the method name, feature order, and method-specific options.
    """
    method = method.lower()
    if method not in avail_methods:
        raise ValueError(f"method must be one of {avail_methods}")

    npz_file, json_file = _npz_and_json_paths(output_file)

    metadata = {"method": method, "features": list(features)}
    arrays = {}

    if method == "standardize":
        arrays["mean"] = np.asarray(scaler.mean_)
        arrays["scale"] = np.asarray(scaler.scale_)
    elif method == "robustize":
        arrays["center"] = np.asarray(scaler.center_)
        arrays["scale"] = np.asarray(scaler.scale_)
    elif method == "mad_robustize":
        arrays["median"] = np.asarray(scaler.median.values)
        arrays["mad"] = np.asarray(scaler.mad.values)
        metadata["epsilon"] = scaler.epsilon
    elif method == "spherize":
        metadata["spherize_method"] = scaler.method
        metadata["center"] = bool(scaler.center)
        metadata["epsilon"] = scaler.epsilon
        arrays["W"] = np.asarray(scaler.W)
        if scaler.method in ["PCA-cor", "ZCA-cor"]:
            arrays["standard_scaler_mean"] = np.asarray(scaler.standard_scaler.mean_)
            arrays["standard_scaler_scale"] = np.asarray(scaler.standard_scaler.scale_)
        elif scaler.center:
            arrays["mean_centerer_mean"] = np.asarray(scaler.mean_centerer.mean_)

    npz_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_file, **arrays)
    with open(json_file, "w") as f:
        json.dump(metadata, f)


def load_normalize_transform(input_file):
    """Load a previously-saved normalize() transform from disk.

    Parameters
    ----------
    input_file : str or pathlib.Path
        Path to the saved transform, as passed to `save_normalize_transform`
        (either the ".npz" file or its companion ".json" file). Both files
        must exist alongside each other.

    Returns
    -------
    scaler : fitted scaler object
        A StandardScaler, RobustScaler, RobustMAD, or Spherize instance with
        its fitted state restored, ready to call `.transform()`.
    method : str
        The normalization method the scaler was fit with.
    features : list of str
        The feature columns (in order) the scaler expects as input.
    """
    npz_file, json_file = _npz_and_json_paths(input_file)

    if not npz_file.exists() or not json_file.exists():
        raise FileNotFoundError(
            f"Could not find both '{npz_file}' and '{json_file}'. "
            "Both files, as written by save_normalize_transform(), are required."
        )

    with open(json_file) as f:
        metadata = json.load(f)

    arrays = np.load(npz_file)
    method = metadata["method"]
    features = metadata["features"]
    n_features = len(features)

    feature_names = np.asarray(features, dtype=object)

    if method == "standardize":
        scaler = StandardScaler()
        scaler.mean_ = arrays["mean"]
        scaler.scale_ = arrays["scale"]
        scaler.var_ = arrays["scale"] ** 2
        scaler.n_features_in_ = n_features
        scaler.feature_names_in_ = feature_names
    elif method == "robustize":
        scaler = RobustScaler()
        scaler.center_ = arrays["center"]
        scaler.scale_ = arrays["scale"]
        scaler.n_features_in_ = n_features
        scaler.feature_names_in_ = feature_names
    elif method == "mad_robustize":
        scaler = RobustMAD(epsilon=metadata["epsilon"])
        scaler.median = pd.Series(arrays["median"], index=features)
        scaler.mad = pd.Series(arrays["mad"], index=features)
    elif method == "spherize":
        scaler = Spherize(
            epsilon=metadata["epsilon"],
            center=metadata["center"],
            method=metadata["spherize_method"],
            return_numpy=True,
        )
        scaler.W = arrays["W"]
        if scaler.method in ["PCA-cor", "ZCA-cor"]:
            standard_scaler = StandardScaler()
            standard_scaler.mean_ = arrays["standard_scaler_mean"]
            standard_scaler.scale_ = arrays["standard_scaler_scale"]
            standard_scaler.var_ = arrays["standard_scaler_scale"] ** 2
            standard_scaler.n_features_in_ = n_features
            scaler.standard_scaler = standard_scaler
        elif scaler.center:
            mean_centerer = StandardScaler(with_mean=True, with_std=False)
            mean_centerer.mean_ = arrays["mean_centerer_mean"]
            mean_centerer.n_features_in_ = n_features
            scaler.mean_centerer = mean_centerer
    else:
        raise ValueError(f"Cannot load transform for unsupported method '{method}'")

    return scaler, method, features
