"""Leakage-safe commodity signal model and walk-forward evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features.financial_features import add_forward_target, add_fusion_features

TECHNICAL=["daily_return","return_5d","return_21d","volatility_5d_ann","volatility_21d_ann",
           "volatility_63d_ann","bollinger_z","rsi_14","macd","macd_signal","momentum_10d",
           "momentum_21d","term_spread"]
MATERIAL=["mqi","mqi_5d_trend","mqi_21d_trend","mqi_63d_trend","supply_disruption_prob",
          "substitution_elasticity","green_premium_per_kg","herfindahl_index"]
INTERACTIONS=["mqi_supply_interaction","quality_adjusted_momentum","recycling_advantage","concentration_shock"]


def backtest(frame: pd.DataFrame, feature_set: str = "full", seed: int = 42) -> tuple[pd.DataFrame,dict[str,float]]:
    """Run expanding 252/21 walk-forward backtest with 10 bps position-change costs."""
    data=add_fusion_features(add_forward_target(frame)).dropna().sort_values("date")
    groups={"technical":TECHNICAL,"material":TECHNICAL+MATERIAL,
            "interaction":TECHNICAL+INTERACTIONS,"full":TECHNICAL+MATERIAL+INTERACTIONS}
    features=groups[feature_set]; outputs=[]
    for commodity,part in data.groupby("commodity"):
        part=part.sort_values("date").reset_index(drop=True)
        for end in range(252,len(part)-20,21):
            train=part.iloc[max(0,end-504):end]; test=part.iloc[end:end+21]
            model=HistGradientBoostingRegressor(max_iter=120,max_depth=4,l2_regularization=.5,
                                                random_state=seed).fit(train[features],train["target_return_21d"])
            # One non-overlapping 21-day decision per rebalance avoids overstating sample size.
            observation=test.iloc[[0]]
            pred=model.predict(observation[features]); position=np.sign(pred)
            pnl=position*observation["target_return_21d"].to_numpy()-0.001*np.abs(position)
            outputs.append(pd.DataFrame({"date":observation["date"],"commodity":commodity,
                "prediction":pred,"actual":observation["target_return_21d"],"position":position,
                "strategy_return":pnl}))
    result=pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame()
    if result.empty: return result,{"sharpe":0.0,"hit_rate":0.0,"max_drawdown":0.0}
    daily=result.groupby("date")["strategy_return"].mean().sort_index()
    sharpe=float(np.sqrt(12)*daily.mean()/daily.std()) if daily.std() else 0.0
    equity=(1+daily).cumprod(); dd=equity/equity.cummax()-1
    return result,{"sharpe":sharpe,"hit_rate":float((np.sign(result.prediction)==np.sign(result.actual)).mean()),
                  "max_drawdown":float(dd.min())}
