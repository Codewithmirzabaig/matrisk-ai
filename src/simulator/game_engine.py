"""Two-level MatRisk Lab campaign state and scoring engine."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GameState:
    level: int=1
    cash: float=1_000_000
    asset_condition: float=7.0
    commodity_position: float=0.0
    insurance_limit: float=500_000
    score: int=0
    turns: int=0


SCENARIOS={1:{"title":"Coastal corrosion alert","shock":.08},
           2:{"title":"Rare-earth supply restriction","shock":.18}}


def apply_decision(state: GameState, maintenance: float, hedge_fraction: float,
                   insurance_limit: float) -> tuple[GameState,dict[str,float|str]]:
    """Advance one turn and score risk-adjusted decisions against an AI benchmark."""
    scenario=SCENARIOS[state.level]; maintenance=max(0,min(250_000,maintenance))
    hedge=max(-1,min(1,hedge_fraction)); insurance=max(0,insurance_limit)
    deterioration=max(0,.9-4*maintenance/1_000_000)
    new_condition=max(1,state.asset_condition-deterioration)
    physical_loss=scenario["shock"]*600_000*(10-new_condition)/9
    insured=min(physical_loss,insurance)
    hedge_pnl=-hedge*scenario["shock"]*250_000
    ending=state.cash-maintenance-physical_loss+insured+hedge_pnl
    risk_bonus=200 if new_condition>=6 else 60
    efficiency=max(0,150-int(maintenance/2000))
    score=state.score+risk_bonus+efficiency+int(max(-100,hedge_pnl/1000))
    next_level=2
    return replace(state,level=next_level,cash=ending,asset_condition=new_condition,
                   commodity_position=hedge,insurance_limit=insurance,score=score,turns=state.turns+1), {
        "scenario":scenario["title"],"physical_loss":physical_loss,"insured_recovery":insured,
        "hedge_pnl":hedge_pnl,"ai_benchmark":"Maintain condition ≥6, insure tail loss, hedge shock exposure"}
