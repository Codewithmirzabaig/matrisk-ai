"""Dependency-light compositional and crystallographic feature generation."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ELEMENT = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


def parse_formula(formula: str) -> dict[str, float]:
    """Parse simple inorganic formulae into element/count mappings."""
    parts = ELEMENT.findall(str(formula))
    if not parts:
        raise ValueError(f"Invalid formula: {formula}")
    return {element: float(count or 1.0) for element, count in parts}


def chemical_system(formula: str) -> str:
    """Create a group-split key so related chemical systems stay in one fold."""
    return "-".join(sorted(parse_formula(formula)))


def featurize_materials(frame: pd.DataFrame) -> pd.DataFrame:
    """Create auditable formula statistics and one-hot crystal descriptors."""
    parsed = frame["formula"].map(parse_formula)
    counts = parsed.map(lambda x: np.array(list(x.values()), dtype=float))
    base = pd.DataFrame({
        "formula_total_atoms": counts.map(np.sum),
        "formula_max_fraction": counts.map(lambda x: x.max()/x.sum()),
        "formula_entropy": counts.map(lambda x: float(-(x/x.sum()*np.log(x/x.sum())).sum())),
    }, index=frame.index)
    numeric = frame.select_dtypes(include="number").drop(columns=[
        "formation_energy_per_atom_eV", "band_gap_eV", "bulk_modulus_GPa", "shear_modulus_GPa",
        "poisson_ratio", "is_stable"], errors="ignore")
    categoricals = pd.get_dummies(frame[["crystal_system", "category"]], dtype=float)
    return pd.concat([numeric, base, categoricals], axis=1).replace([np.inf,-np.inf], np.nan).fillna(0)

