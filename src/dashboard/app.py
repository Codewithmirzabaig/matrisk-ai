"""Six-view MatRisk decision cockpit, including the two-level Lab simulator."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data.loaders import load_csv, merge_market_data
from src.financial.credit_risk import calculate_credit_risk
from src.financial.esg_module import substitution_impact
from src.financial.stress_testing import SCENARIOS, run_stress
from src.models.gan.inverse_designer import generate_candidates
from src.models.survival.risk import bridge_hazard, survival_curve
from src.simulator.game_engine import GameState, apply_decision

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/"data"/"raw"
st.set_page_config(page_title="MatRisk AI",page_icon="◈",layout="wide")
st.markdown("""<style>.block-container{padding-top:1.5rem}.metric-card{border:1px solid #26334a;border-radius:14px;padding:12px}</style>""",unsafe_allow_html=True)
st.title("MatRisk AI · Material intelligence for financial risk")
page=st.sidebar.radio("Decision workspace",["Material Explorer","Commodity Signals","Infrastructure Risk","Inverse Designer","ESG Impact","MatRisk Lab"])

@st.cache_data
def data(name): return load_csv(DATA/name)

if page=="Material Explorer":
    df=data("DS1_material_properties_5500.csv"); formula=st.text_input("Chemical formula",df.formula.iloc[0])
    selected=df[df.formula.str.lower()==formula.lower()]
    if selected.empty: selected=df.iloc[[0]]; st.info("Formula not in sample; showing nearest category baseline.")
    row=selected.iloc[0]; cols=st.columns(4)
    for c,(label,key,unit) in zip(cols,[("Formation energy","formation_energy_per_atom_eV","eV/atom"),("Band gap","band_gap_eV","eV"),("Bulk modulus","bulk_modulus_GPa","GPa"),("Shear modulus","shear_modulus_GPa","GPa")]): c.metric(label,f"{row[key]:.2f} {unit}")
    peers=df[df.crystal_system==row.crystal_system].assign(distance=lambda x:abs(x.density_g_cm3-row.density_g_cm3)).nsmallest(5,"distance")
    st.subheader("Five structurally similar materials"); st.dataframe(peers[["formula","crystal_system","category","density_g_cm3","bulk_modulus_GPa","shear_modulus_GPa"]],use_container_width=True)
    st.plotly_chart(px.scatter(df.sample(min(1200,len(df)),random_state=42),x="bulk_modulus_GPa",y="shear_modulus_GPa",color="crystal_system",hover_data=["formula"]),use_container_width=True)
elif page=="Commodity Signals":
    merged=merge_market_data(data("DS2_commodity_prices_10yr.csv"),data("DS4_crossdomain_features_daily.csv")); commodity=st.selectbox("Commodity",sorted(merged.commodity.unique()))
    d=merged[merged.commodity==commodity].sort_values("date").tail(300); a,b,c=st.columns(3); a.metric("MQI",f"{d.mqi.iloc[-1]:.2f}",f"{d.mqi_21d_trend.iloc[-1]:+.2f}"); b.metric("Supply disruption",f"{d.supply_disruption_prob.iloc[-1]:.1%}"); c.metric("Substitution elasticity",f"{d.substitution_elasticity.iloc[-1]:.2f}")
    fig=go.Figure(go.Candlestick(x=d.date,open=d.open,high=d.high,low=d.low,close=d.close,name=commodity)); fig.add_trace(go.Scatter(x=d.date,y=d.sma_21,name="SMA 21")); st.plotly_chart(fig,use_container_width=True)
    forecast=d.close.iloc[-1]*(1+d.mqi_21d_trend.iloc[-1]/100); st.metric("Material-adjusted 21-day scenario",f"{forecast:,.2f}",help="Scenario indicator, not investment advice")
elif page=="Infrastructure Risk":
    df=data("DS3_infrastructure_bridges_5000.csv"); bridge=st.selectbox("Bridge",df.bridge_id.astype(str)); row=df[df.bridge_id.astype(str)==bridge].iloc[[0]]; hazard=float(bridge_hazard(row)[0]); curve=survival_curve(hazard); credit=calculate_credit_risk(float(curve.loc[curve.year==5,"pd"].iloc[0]),row.loan_outstanding_M.iloc[0],row.condition_rating.iloc[0]);
    a,b,c=st.columns(3); a.metric("5-year PD",f"{credit.pd:.1%}"); b.metric("LGD",f"{credit.lgd:.1%}"); c.metric("Expected loss",f"${credit.expected_loss:.2f}M"); st.plotly_chart(px.line(curve,x="year",y=["survival","pd"],title="Survival and cumulative default probability"),use_container_width=True)
    scenario=st.selectbox("Stress scenario",list(SCENARIOS)); st.json(run_stress(scenario,row.replacement_cost_M.iloc[0]*1e6,paths=10000))
elif page=="Inverse Designer":
    prices=data("DS5_element_prices_monthly.csv"); c1,c2,c3=st.columns(3); strength=c1.slider("Target strength (MPa)",200,1200,700); density=c2.slider("Target density (g/cm³)",2.0,10.0,6.0); budget=c3.slider("Budget ($/kg)",5,500,100)
    candidates=generate_candidates(prices,budget,strength,density); st.dataframe(candidates,use_container_width=True); st.plotly_chart(px.scatter(candidates,x="cost_usd_kg",y="property_score",size="predicted_strength_MPa",color="within_budget",hover_name="composition",title="Cost-property Pareto screen"),use_container_width=True)
elif page=="ESG Impact":
    fusion=data("DS4_crossdomain_features_daily.csv"); commodities=sorted(fusion.commodity.unique()); original=st.selectbox("Original material",commodities); substitute=st.selectbox("Substitute",[x for x in commodities if x!=original]); share=st.slider("Recycled/substitute share",0.,1.,.5)
    o=fusion[fusion.commodity==original].iloc[-1]; s=fusion[fusion.commodity==substitute].iloc[-1]; impact=substitution_impact(o.carbon_intensity_virgin,s.carbon_intensity_recycled,share,100,100+s.green_premium_per_kg)
    a,b,c=st.columns(3); a.metric("Carbon reduction",f"{impact['carbon_reduction_pct']:.1f}%"); b.metric("Cost delta",f"${impact['cost_delta']:.2f}/kg"); c.metric("Green bond screen","Eligible" if impact["green_bond_eligible"] else "Review")
    st.plotly_chart(px.bar(pd.DataFrame({"case":["Before","After"],"carbon":[impact["carbon_before"],impact["carbon_after"]]}),x="case",y="carbon",color="case"),use_container_width=True)
else:
    if "game" not in st.session_state: st.session_state.game=GameState()
    state=st.session_state.game; st.subheader(f"Campaign Level {state.level}: risk under uncertainty"); a,b,c=st.columns(3); a.metric("Portfolio cash",f"${state.cash:,.0f}"); b.metric("Asset condition",f"{state.asset_condition:.1f}/9"); c.metric("Score",state.score)
    maintenance=st.slider("Preventive maintenance",0,250000,75000,5000); hedge=st.slider("Commodity hedge",-1.,1.,-.4,.1); insurance=st.slider("Insurance limit",0,1000000,500000,25000)
    if st.button("Commit decision",type="primary"):
        st.session_state.game,outcome=apply_decision(state,maintenance,hedge,insurance); st.success(f"Turn complete: {outcome['scenario']}"); st.json(outcome); st.rerun()

st.caption("Decision-support prototype · estimates are scenario outputs, not investment, credit, or engineering advice")

