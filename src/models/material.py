"""Group-aware multi-output material property baseline with uncertainty."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from src.features.material_features import chemical_system, featurize_materials

TARGETS = ["formation_energy_per_atom_eV", "band_gap_eV", "bulk_modulus_GPa", "shear_modulus_GPa"]


@dataclass
class MaterialModelBundle:
    """Model, schema, and holdout metrics needed for reproducible inference."""
    model: ExtraTreesRegressor
    feature_names: list[str]
    metrics: dict[str, dict[str, float]]


def train_material_model(frame: pd.DataFrame, seed: int = 42, n_estimators: int = 300) -> MaterialModelBundle:
    """Train using chemical-system groups to prevent family leakage."""
    x = featurize_materials(frame)
    y = frame[TARGETS]
    groups = frame["formula"].map(chemical_system)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    model = ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=2,
                               random_state=seed, n_jobs=-1)
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(x.iloc[test_idx])
    metrics = {target: {
        "mae": float(mean_absolute_error(y.iloc[test_idx, i], pred[:, i])),
        "r2": float(r2_score(y.iloc[test_idx, i], pred[:, i])),
    } for i, target in enumerate(TARGETS)}
    return MaterialModelBundle(model, list(x.columns), metrics)


def predict_with_uncertainty(bundle: MaterialModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """Return ensemble mean and empirical 90% interval across trees."""
    x = featurize_materials(frame).reindex(columns=bundle.feature_names, fill_value=0)
    draws = np.stack([tree.predict(x) for tree in bundle.model.estimators_])
    out: dict[str, np.ndarray] = {}
    for i, target in enumerate(TARGETS):
        out[target] = draws[:, :, i].mean(axis=0)
        out[f"{target}_p05"] = np.quantile(draws[:, :, i], .05, axis=0)
        out[f"{target}_p95"] = np.quantile(draws[:, :, i], .95, axis=0)
    return pd.DataFrame(out, index=frame.index)

