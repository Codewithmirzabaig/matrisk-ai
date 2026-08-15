import numpy as np
import pandas as pd
from src.data.loaders import elastic_consistency, merge_market_data
from src.features.material_features import parse_formula, chemical_system
from src.features.financial_features import chronological_split
from src.models.pinn.physics_losses import project_elastic, physics_audit
from src.financial.credit_risk import calculate_credit_risk
from src.financial.cat_model import simulate_aggregate_loss, tail_metrics
from src.financial.esg_module import substitution_impact
from src.financial.stress_testing import run_stress
from src.simulator.game_engine import GameState, apply_decision

def test_formula_and_group():
    assert parse_formula("Fe2O3")=={"Fe":2.0,"O":3.0}
    assert chemical_system("Fe2O3")==chemical_system("O3Fe2")=="Fe-O"

def test_elastic_projection_is_consistent():
    k,g,e=project_elastic(np.array([150.]),np.array([80.]))
    nu=(3*k-2*g)/(2*(3*k+g))
    assert elastic_consistency(k,g,nu)[0] < 1e-12
    assert physics_audit(k,g,nu)["all_constraints"]==1.0

def test_point_in_time_merge_and_split():
    dates=pd.date_range("2024-01-01",periods=10)
    a=pd.DataFrame({"date":dates,"commodity":"X","close":range(10)})
    b=pd.DataFrame({"date":dates,"commodity":"X","mqi":range(10)})
    merged=merge_market_data(a,b); train,test=chronological_split(merged,.8)
    assert len(merged)==10 and train.date.max()<test.date.min()

def test_credit_identity_and_bounds():
    r=calculate_credit_risk(.2,10,6)
    assert r.expected_loss==r.pd*r.lgd*r.ead
    assert calculate_credit_risk(3,-2,9).pd==1 and calculate_credit_risk(3,-2,9).ead==0

def test_cat_tail_ordering():
    losses=simulate_aggregate_loss(np.array([.1,.2]),np.array([1e6,2e6]),1000)
    m=tail_metrics(losses)
    assert len(losses)==1000 and m["tvar"]>=m["var"]>=m["expected_loss"]

def test_esg_and_stress_are_deterministic():
    impact=substitution_impact(10,2,.5,100,110)
    assert impact["carbon_reduction_pct"]==40 and impact["green_bond_eligible"]
    assert run_stress("Climate Corrosion",1e6,100,seed=7)==run_stress("Climate Corrosion",1e6,100,seed=7)

def test_game_advances_and_preserves_bounds():
    state,outcome=apply_decision(GameState(),75000,-.5,500000)
    assert state.turns==1 and 1<=state.asset_condition<=9 and "ai_benchmark" in outcome

