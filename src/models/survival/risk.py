"""Transparent infrastructure survival proxy calibrated to bridge condition data."""
from __future__ import annotations
import numpy as np
import pandas as pd


def bridge_hazard(frame: pd.DataFrame) -> np.ndarray:
    """Estimate annual hazard from age, condition, corrosion, fatigue, and section loss."""
    age_ratio = frame["age_years"] / frame["design_life_years"].clip(lower=1)
    condition = (9-frame["condition_rating"])/8
    loss = 1-frame["remaining_thickness_mm"]/frame["original_thickness_mm"].clip(lower=1e-6)
    fatigue = frame["fatigue_cycles_millions"] / frame["fatigue_cycles_millions"].quantile(.95)
    logit = -5 + 2.1*age_ratio + 2.6*condition + 6*frame["corrosion_rate_mm_yr"] + 1.5*loss + .4*fatigue
    return np.clip(1/(1+np.exp(-logit)), 1e-5, .95)


def survival_curve(annual_hazard: float, years: int = 30) -> pd.DataFrame:
    """Convert constant annual hazard into survival and cumulative PD curves."""
    t=np.arange(0, years+1)
    survival=np.power(1-float(annual_hazard), t)
    return pd.DataFrame({"year":t,"survival":survival,"pd":1-survival})

