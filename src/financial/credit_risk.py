"""PD/LGD/EAD/EL calculations with explicit formulas."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CreditRisk:
    pd: float
    lgd: float
    ead: float
    expected_loss: float


def calculate_credit_risk(pd: float, loan_outstanding_m: float, condition_rating: float,
                          collateral_recovery: float = .35) -> CreditRisk:
    """Calculate expected loss in millions from bounded PD, LGD, and EAD."""
    probability=max(0.0,min(1.0,float(pd)))
    recovery=max(0.0,min(1.0,collateral_recovery*(condition_rating/9)))
    lgd=1-recovery
    ead=max(0.0,float(loan_outstanding_m))
    return CreditRisk(probability,lgd,ead,probability*lgd*ead)

