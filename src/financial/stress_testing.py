"""Five-scenario Monte Carlo material-financial stress engine."""
from __future__ import annotations

import numpy as np

SCENARIOS={
 "Rare Earth Crisis":(-.18,.24), "Steel Quality Shift":(-.10,.14),
 "Climate Corrosion":(-.14,.18), "Recycled Mandate":(-.06,.11),
 "Novel Alloy Disruption":(.09,.16),
}


def run_stress(scenario: str, portfolio_value: float, paths: int = 10000,
               severity: float = 1.0, seed: int = 42) -> dict[str,float]:
    """Simulate portfolio P&L and return downside risk metrics."""
    if scenario not in SCENARIOS: raise KeyError(f"Unknown scenario: {scenario}")
    mean,vol=SCENARIOS[scenario]
    rng=np.random.default_rng(seed)
    pnl=portfolio_value*rng.normal(mean*severity,vol*np.sqrt(severity),paths)
    q95,q99=np.quantile(pnl,[.05,.01]); cutoff=np.quantile(pnl,.005)
    return {"expected_pnl":float(pnl.mean()),"var_95":float(-q95),"var_99":float(-q99),
            "tvar_99_5":float(-pnl[pnl<=cutoff].mean()),"maximum_loss":float(-pnl.min())}

