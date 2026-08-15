"""Physics constraints usable by both neural and tabular prediction layers."""
from __future__ import annotations

import numpy as np


def elastic_loss(k: np.ndarray, g: np.ndarray, nu: np.ndarray) -> float:
    """Mean squared discrepancy between two equivalent Young's modulus equations."""
    e_k = 3*np.asarray(k)*(1-2*np.asarray(nu))
    e_g = 2*np.asarray(g)*(1+np.asarray(nu))
    scale = np.maximum(np.abs(e_k), 1.0)
    return float(np.mean(((e_k-e_g)/scale)**2))


def project_elastic(k: np.ndarray, g: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Project positive K/G predictions onto an isotropic, physically consistent manifold."""
    k = np.maximum(np.asarray(k, dtype=float), 1e-6)
    g = np.maximum(np.asarray(g, dtype=float), 1e-6)
    nu = np.clip((3*k-2*g)/(2*(3*k+g)), -0.999, 0.499)
    young = 2*g*(1+nu)
    return k, g, young


def physics_audit(k: np.ndarray, g: np.ndarray, nu: np.ndarray, tolerance: float = .10) -> dict[str,float]:
    """Report bound and elastic-consistency compliance rates."""
    k, g, nu = map(np.asarray, (k,g,nu))
    ek, eg = 3*k*(1-2*nu), 2*g*(1+nu)
    consistent = np.abs(ek-eg)/np.maximum(np.abs(ek), 1e-9) <= tolerance
    valid = (k>0)&(g>0)&(nu>-1)&(nu<.5)&consistent
    return {"positive_moduli": float(((k>0)&(g>0)).mean()),
            "poisson_bounds": float(((nu>-1)&(nu<.5)).mean()),
            "elastic_consistency": float(consistent.mean()), "all_constraints": float(valid.mean())}

