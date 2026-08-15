"""Compound-Poisson catastrophe simulation and tail-risk metrics."""
from __future__ import annotations

import numpy as np


def simulate_aggregate_loss(failure_probabilities: np.ndarray, replacement_values: np.ndarray,
                            simulations: int = 10000, seed: int = 42) -> np.ndarray:
    """Simulate correlated-free annual events with log-normal conditional severity."""
    p=np.clip(np.asarray(failure_probabilities,float),0,1)
    v=np.maximum(np.asarray(replacement_values,float),0)
    if p.shape != v.shape: raise ValueError("probabilities and values must align")
    rng=np.random.default_rng(seed)
    events=rng.random((simulations,len(p))) < p
    severity=np.clip(rng.lognormal(mean=np.log(.35),sigma=.55,size=events.shape),0,1.5)
    return (events*severity*v).sum(axis=1)


def tail_metrics(losses: np.ndarray, quantile: float = .995) -> dict[str,float]:
    """Calculate mean loss, VaR, TVaR, and worst case from simulations."""
    values=np.asarray(losses,float)
    var=float(np.quantile(values,quantile))
    tail=values[values>=var]
    return {"expected_loss":float(values.mean()),"var":var,
            "tvar":float(tail.mean()) if len(tail) else var,"maximum_loss":float(values.max())}

