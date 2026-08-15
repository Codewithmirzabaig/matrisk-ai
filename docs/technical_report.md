# MatRisk AI — Technical Report

## 1. Introduction

Financial institutions price material-dependent assets with abstractions that often omit the physics
governing degradation, quality, substitution, and failure. MatRisk AI tests a convergence thesis: material
quality and physical degradation indicators can improve commodity scenarios and make infrastructure
risk translation more explainable. The system contributes an auditable cross-domain feature layer, a
physics compliance gate, financial translations, and a playable decision environment.

Research questions are: (RQ1) can compositional and structural attributes predict four material
properties without leaking related chemical families across evaluation splits; (RQ2) do MQI and supply
signals add out-of-time commodity information beyond technical features; (RQ3) can physical degradation
be translated into PD, LGD, expected loss, and insurance TVaR transparently; and (RQ4) can interactive
scenarios teach the resulting trade-offs?

## 2. Literature review

Crystal graph methods learn representations by passing messages between atomic neighbours. CGCNN
(Xie and Grossman, 2018), SchNet (Schütt et al., 2018), MEGNet (Chen et al., 2019), and ALIGNN
(Choudhary and DeCost, 2021) established strong property-prediction baselines. Physics-informed learning
adds governing equations to the objective or projects outputs onto valid manifolds (Raissi et al., 2019;
Karniadakis et al., 2021). Alternative-data studies show that non-price signals can improve financial
models only when timestamps, publication lags, and backtests are controlled (Lopez de Prado, 2018).
Infrastructure reliability combines deterioration, survival analysis, and inspection evidence (Cox, 1972;
FHWA, 2024). Tail risk uses loss-frequency and severity models rather than mean loss alone (McNeil,
Frey, and Embrechts, 2015).

## 3. Data and methodology

The six supplied files contain 58,684 rows. DS1 supports material modelling; DS2 and DS4 join exactly on
date and commodity; DS3 contains bridge state and exposure; DS5 provides monthly element costs; DS6
contains failure severity. Automated gates check completeness, duplication, physical bounds, unique join
keys, and chronological ordering.

### 3.1 Leakage controls

For material prediction, formulas are mapped to sorted chemical systems and passed to a group split, so
all compounds in the same elemental family remain on one side of the holdout boundary. For commodity
prediction, the target is the forward 21-day return and models train only on rows dated before each test
window. Rolling nulls are treated as unavailable warm-up observations, not imputed with future values.

### 3.2 Material and physics layer

The sample-compatible baseline is a multi-output Extra Trees ensemble. Given features \(x\), each tree
predicts \(\hat y_b(x)\); the ensemble estimate and uncertainty are the mean and empirical 5th/95th
percentiles across trees. A future CGNN adapter can replace this baseline once atomic coordinates exist.

For isotropic elasticity,

\[
E_K=3K(1-2\nu), \qquad E_G=2G(1+\nu).
\]

The normalized physics loss is \(L_p=\mathbb{E}[(E_K-E_G)^2/\max(|E_K|,1)^2]\). The audit also requires
\(K>0\), \(G>0\), and \(-1<\nu<0.5\). Projection computes
\(\nu=(3K-2G)/(2(3K+G))\), which enforces elastic consistency for positive moduli.

### 3.3 Commodity evaluation

Four feature variants support ablation: technical only; technical plus material; technical plus fusion
interactions; and full. An expanding walk-forward model begins after 252 observations, retrains every 21
days, and charges 10 basis points on position changes. Outputs are annualized Sharpe, directional hit
rate, and maximum drawdown. This is a research backtest, not evidence of tradable performance.

### 3.4 Infrastructure and financial translation

The prototype hazard is a bounded, monotonic function of age/design-life ratio, inverse condition,
corrosion, thickness loss, and fatigue. For annual hazard \(h\), survival is \(S(t)=(1-h)^t\) and
\(PD(t)=1-S(t)\). Credit expected loss is

\[
EL=PD\times LGD\times EAD.
\]

Insurance aggregate loss is generated from Bernoulli failures and bounded log-normal conditional
severity. VaR is a loss quantile and TVaR is mean loss beyond that quantile. Five deterministic-seed
stress families return expected P&L, 95%/99% VaR, 99.5% TVaR, and maximum loss.

### 3.5 Inverse design and ESG

Candidate element fractions are sampled from a Dirichlet simplex, guaranteeing non-negativity and a
unit sum. Candidates are screened against the latest element cost vector and ranked by normalized target
distance. Because the supplied DS1 formulas do not include atomic structures or calibrated elemental
property tensors, property estimates are clearly labelled screening surrogates rather than CGNN claims.
The ESG module compares blended carbon intensity, cost delta, and a configurable eligibility screen.

## 4. Evaluation protocol

`scripts/run_quality.py` records completeness and physical validity. `scripts/train.py` persists a model
bundle and held-out MAE/R² for all four property targets. The commodity ablation is run from source data;
results must not be copied across runs because environment versions and data revisions can change them.
Tests verify algebraic identities, point-in-time order, deterministic simulation, simplex constraints,
tail ordering, and game-state transitions. CI repeats lint and coverage checks on every push.

Benchmark gaps must be reported, not hidden. The project brief's literature-scale CGNN thresholds are
unlikely to be defensible on 5,500 tabular records without atomic structures. Model promotion should
require external crystal data, nested group cross-validation, uncertainty calibration, and registered
experiment lineage.

## 5. Discussion, limitations, and model risk

The strongest aspect is integration: a reviewer can trace material state into an explicit financial metric
and inspect every assumption. The largest limitation is synthetic/sample data. Hazard coefficients are
expert-designed and require calibration to longitudinal inspection/failure histories. Commodity features
may be contemporaneously generated and must be checked for publication time before live use. Element
property surrogates are not suitable for laboratory decisions. Independent engineering assessment remains
mandatory.

Failure modes include regime change, feature timestamp errors, family leakage, overconfident tree
intervals, dependency drift, and users treating scenarios as forecasts. Production controls should include
data contracts, challenger models, calibration dashboards, drift alerts, approvals, audit logs, and defined
fallbacks.

## 6. Conclusion and roadmap

MatRisk AI demonstrates a coherent material-to-finance architecture without overstating what the sample
data can prove. Next steps are ingestion of Materials Project/AFLOW structures and full NBI inspection
panels; CGNN and DeepSurv training; paired-bootstrap commodity ablation; WGAN-GP training; MLflow
registration; and independent model validation. Real-time sensor fusion and portfolio optimization follow
only after these gates pass.

## References

1. Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2).
2. Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks. *Physical Review Letters*, 120.
3. Schütt, K. T., et al. (2018). SchNet. *Journal of Chemical Physics*, 148.
4. Chen, C., et al. (2019). Graph networks as a universal ML framework for molecules and crystals. *Chemistry of Materials*, 31.
5. Choudhary, K., & DeCost, B. (2021). Atomistic line graph neural network. *npj Computational Materials*, 7.
6. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378.
7. Karniadakis, G. E., et al. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3.
8. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
9. McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management*. Princeton.
10. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.
11. Breiman, L. (2001). Random forests. *Machine Learning*, 45.
12. Friedman, J. H. (2001). Greedy function approximation. *Annals of Statistics*, 29.
13. Goodfellow, I., et al. (2014). Generative adversarial nets. NeurIPS.
14. Gulrajani, I., et al. (2017). Improved training of Wasserstein GANs. NeurIPS.
15. Katzman, J. L., et al. (2018). DeepSurv. *BMC Medical Research Methodology*, 18.
16. Lundberg, S. M., et al. (2020). From local explanations to global understanding. *Nature Machine Intelligence*, 2.
17. National Academies (2019). *Fracture, Fatigue, Failure, and Damage Evolution*. NASEM.
18. FHWA (2024). *National Bridge Inventory Coding Guide*. U.S. DOT.
19. Basel Committee (2019). *Minimum Capital Requirements for Market Risk*. BIS.
20. IAIS (2020). *Application Paper on the Supervision of Climate-related Risks*. IAIS.
21. NIST (2023). *AI Risk Management Framework 1.0*. U.S. Department of Commerce.
22. Sculley, D., et al. (2015). Hidden technical debt in ML systems. NeurIPS.

