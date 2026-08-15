import numpy as np
import pandas as pd

from src.common import load_config, seed_everything
from src.data.loaders import (
    elastic_consistency,
    load_csv,
    merge_market_data,
    validate_materials,
)
from src.features.financial_features import (
    add_forward_target,
    add_fusion_features,
    chronological_split,
)
from src.features.material_features import (
    chemical_system,
    featurize_materials,
    parse_formula,
)
from src.financial.cat_model import simulate_aggregate_loss, tail_metrics
from src.financial.commodity_predictor import MATERIAL, TECHNICAL, backtest
from src.financial.credit_risk import calculate_credit_risk
from src.financial.esg_module import substitution_impact
from src.financial.stress_testing import run_stress
from src.models.gan.inverse_designer import generate_candidates
from src.models.material import predict_with_uncertainty, train_material_model
from src.models.pinn.physics_losses import physics_audit, project_elastic
from src.models.survival.risk import bridge_hazard, survival_curve
from src.simulator.game_engine import GameState, apply_decision


def test_formula_and_group():
    assert parse_formula("Fe2O3")=={"Fe":2.0,"O":3.0}
    assert chemical_system("Fe2O3")==chemical_system("O3Fe2")=="Fe-O"

def test_elastic_projection_is_consistent():
    k,g,_e=project_elastic(np.array([150.]),np.array([80.]))
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


def test_config_seed_and_csv_loader(tmp_path):
    config = load_config()
    assert config["random_seed"] == 42
    seed_everything(7)
    first = np.random.random()
    seed_everything(7)
    assert np.random.random() == first
    path = tmp_path / "sample.csv"
    path.write_text("date,value\n2024-01-01,1\n", encoding="utf-8")
    assert pd.api.types.is_datetime64_any_dtype(load_csv(path)["date"])


def test_material_validation_and_features():
    frame = pd.DataFrame(
        {
            "formula": ["Fe2O3", "Al2O3"],
            "crystal_system": ["cubic", "hexagonal"],
            "category": ["oxide", "oxide"],
            "spacegroup_number": [1, 2],
            "n_elements": [2, 2],
            "bulk_modulus_GPa": [150.0, 200.0],
            "shear_modulus_GPa": [80.0, 100.0],
            "poisson_ratio": [0.30, 0.28],
            "density_g_cm3": [5.2, 4.0],
            "formation_energy_per_atom_eV": [-1.0, -2.0],
            "band_gap_eV": [2.0, 4.0],
            "is_stable": [1, 1],
        }
    )
    assert validate_materials(frame).passed
    assert featurize_materials(frame).shape[0] == 2
    broken = frame.copy()
    broken.loc[0, "density_g_cm3"] = -1
    assert not validate_materials(broken).passed
    assert "non-positive density" in validate_materials(broken).violations
    try:
        parse_formula("not-a-formula")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid formula accepted")


def test_financial_feature_generation():
    dates = pd.date_range("2024-01-01", periods=25)
    frame = pd.DataFrame(
        {
            "date": dates,
            "commodity": "Copper",
            "close": np.arange(100.0, 125.0),
            "mqi": 1.0,
            "supply_disruption_prob": 0.2,
            "momentum_21d": 0.1,
            "mqi_21d_trend": 0.05,
            "carbon_intensity_virgin": 10.0,
            "carbon_intensity_recycled": 3.0,
            "herfindahl_index": 0.4,
        }
    )
    targeted = add_forward_target(frame, horizon=2)
    assert targeted["target_return_21d"].notna().sum() == 23
    fused = add_fusion_features(targeted)
    assert np.isclose(fused["recycling_advantage"].iloc[0], 0.7)


def test_survival_credit_and_inverse_design():
    bridges = pd.DataFrame(
        {
            "age_years": [50], "design_life_years": [75], "condition_rating": [5],
            "remaining_thickness_mm": [12.0], "original_thickness_mm": [18.0],
            "corrosion_rate_mm_yr": [0.08], "fatigue_cycles_millions": [2.0],
        }
    )
    hazard = bridge_hazard(bridges)[0]
    curve = survival_curve(hazard, years=5)
    assert 0 < hazard < 1 and curve["pd"].is_monotonic_increasing
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "element": ["Fe", "Al", "Cu", "Ni"],
            "price_usd_per_kg": [1.0, 2.0, 8.0, 15.0],
        }
    )
    candidates = generate_candidates(prices, 20, 700, 6, n=3, trials=100)
    assert len(candidates) == 3
    assert np.allclose(candidates["fractions_sum"], 1.0)


def test_material_model_training_and_uncertainty():
    rows = []
    formulas = ["FeO", "AlO", "CuO", "NiO", "TiO", "ZnO", "MgO", "CaO", "SiO", "BO"]
    for index, formula in enumerate(formulas * 3):
        rows.append(
            {
                "formula": formula,
                "crystal_system": "cubic" if index % 2 else "hexagonal",
                "category": "oxide",
                "n_elements": 2,
                "spacegroup_number": index + 1,
                "density_g_cm3": 2 + index / 10,
                "formation_energy_per_atom_eV": -1 - index / 100,
                "band_gap_eV": index / 20,
                "bulk_modulus_GPa": 100 + index,
                "shear_modulus_GPa": 50 + index / 2,
                "poisson_ratio": 0.3,
                "is_stable": 1,
            }
        )
    frame = pd.DataFrame(rows)
    bundle = train_material_model(frame, n_estimators=10)
    prediction = predict_with_uncertainty(bundle, frame.iloc[:2])
    assert set(bundle.metrics) == {
        "formation_energy_per_atom_eV", "band_gap_eV", "bulk_modulus_GPa", "shear_modulus_GPa"
    }
    assert prediction.shape == (2, 12)


def test_commodity_backtest_empty_and_validation_errors():
    short = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10),
                          "commodity": "Copper", "close": np.arange(100.0, 110.0)})
    for column in set(TECHNICAL + MATERIAL):
        short[column] = 0.1
    short["carbon_intensity_virgin"] = 10.0
    short["carbon_intensity_recycled"] = 3.0
    empty, metrics = backtest(short, "full")
    assert empty.empty and metrics["sharpe"] == 0
    try:
        run_stress("Unknown", 1_000)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown scenario accepted")
    try:
        simulate_aggregate_loss(np.array([0.1]), np.array([1.0, 2.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned loss vectors accepted")
