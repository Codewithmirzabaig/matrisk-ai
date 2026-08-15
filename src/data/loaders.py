"""Typed loaders and data-quality gates for the supplied MatRisk datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityResult:
    """Compact, serialisable dataset quality result."""

    rows: int
    columns: int
    completeness: float
    duplicate_rows: int
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.completeness >= 0.80 and not self.violations


def load_csv(path: str | Path, dates: tuple[str, ...] = ("date", "last_inspection_date")) -> pd.DataFrame:
    """Load a CSV and parse any known date columns without mutating the source."""
    frame = pd.read_csv(path)
    for column in dates:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def validate_materials(frame: pd.DataFrame) -> QualityResult:
    """Apply the physical plausibility checks specified in the project brief."""
    required = {"bulk_modulus_GPa", "shear_modulus_GPa", "poisson_ratio", "density_g_cm3"}
    missing = required.difference(frame.columns)
    issues: list[str] = []
    if missing:
        issues.append(f"missing columns: {sorted(missing)}")
    else:
        if (frame["bulk_modulus_GPa"] <= 0).any(): issues.append("non-positive bulk modulus")
        if (frame["shear_modulus_GPa"] <= 0).any(): issues.append("non-positive shear modulus")
        if (~frame["poisson_ratio"].between(-1, 0.5, inclusive="neither")).any():
            issues.append("Poisson ratio outside (-1, 0.5)")
        if (frame["density_g_cm3"] <= 0).any(): issues.append("non-positive density")
    return QualityResult(len(frame), len(frame.columns), float(1-frame.isna().mean().mean()),
                         int(frame.duplicated().sum()), tuple(issues))


def elastic_consistency(k: pd.Series | np.ndarray, g: pd.Series | np.ndarray,
                        nu: pd.Series | np.ndarray) -> np.ndarray:
    """Return relative inconsistency between Young's modulus derived from K and G."""
    k, g, nu = np.asarray(k), np.asarray(g), np.asarray(nu)
    e_k = 3 * k * (1 - 2 * nu)
    e_g = 2 * g * (1 + nu)
    return np.abs(e_k - e_g) / np.maximum(np.abs(e_k), 1e-9)


def merge_market_data(prices: pd.DataFrame, fusion: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time merge on the exact (date, commodity) key."""
    keys = ["date", "commodity"]
    if prices.duplicated(keys).any() or fusion.duplicated(keys).any():
        raise ValueError("Non-unique point-in-time market key")
    merged = prices.merge(fusion, on=keys, how="inner", validate="one_to_one")
    return merged.sort_values(keys, kind="stable").reset_index(drop=True)

