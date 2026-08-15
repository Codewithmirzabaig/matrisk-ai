"""Material-substitution carbon, cost, and green-bond screening."""
from __future__ import annotations


def substitution_impact(virgin_carbon: float, recycled_carbon: float, recycled_share: float,
                        cost_before: float, cost_after: float) -> dict[str,float|bool]:
    """Evaluate carbon reduction and a transparent green-bond eligibility rule."""
    share=max(0.0,min(1.0,recycled_share))
    before=max(float(virgin_carbon),1e-9)
    after=(1-share)*before+share*max(float(recycled_carbon),0)
    reduction=1-after/before
    cost_delta=float(cost_after)-float(cost_before)
    return {"carbon_before":before,"carbon_after":after,"carbon_reduction_pct":100*reduction,
            "cost_delta":cost_delta,"green_bond_eligible":reduction>=.30 and cost_delta/cost_before<=.20}

