"""Point-in-time financial and cross-domain fusion features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_target(frame: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Create a forward return target only after all contemporaneous features exist."""
    ordered = frame.sort_values(["commodity", "date"], kind="stable").copy()
    ordered["target_return_21d"] = ordered.groupby("commodity")["close"].transform(
        lambda s: s.shift(-horizon) / s - 1
    )
    return ordered


def add_fusion_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate cross-domain interactions specified by the MatRisk thesis."""
    result = frame.copy()
    result["mqi_supply_interaction"] = result["mqi"] * result["supply_disruption_prob"]
    result["quality_adjusted_momentum"] = result["momentum_21d"] * result["mqi_21d_trend"]
    result["recycling_advantage"] = (
        result["carbon_intensity_virgin"] - result["carbon_intensity_recycled"]
    ) / result["carbon_intensity_virgin"].clip(lower=1e-9)
    result["concentration_shock"] = result["herfindahl_index"] * result["supply_disruption_prob"]
    return result.replace([np.inf, -np.inf], np.nan)


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Split on a date boundary; never randomly mix future observations into training."""
    dates = np.sort(frame["date"].dropna().unique())
    cutoff = dates[max(1, int(len(dates)*train_fraction))-1]
    return frame[frame["date"] <= cutoff].copy(), frame[frame["date"] > cutoff].copy()

