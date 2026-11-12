# Motor Insurance Pricing: Claim Frequency Model

Predicting the expected number of claims per policy per year on a French motor insurance portfolio, comparing a Poisson GLM with LightGBM.

## Why it matters
Insurers price a policy as **expected claim frequency × average claim cost**. This project focuses on the frequency part: if a model ranks drivers by risk more accurately, the insurer can offer lower prices to safe drivers while charging enough to cover riskier ones.

## Data
[freMTPL2freq](https://www.openml.org/d/41214): 678,013 French motor third-party liability policies with driver, vehicle and location features, claim counts and exposure (policy duration in years).

## Approach
1. **Exploratory analysis** of claim frequency by risk factor, weighting by exposure.
2. **Data preparation:** capping of extreme values, log-density, age bands for the GLM, and an interaction feature found during the analysis.
3. **Poisson GLM** (scikit-learn) with exposure as sample weight.
4. **LightGBM** with a Poisson objective and early stopping.
5. **Evaluation** on a held-out test set with Poisson deviance, calibration and Lorenz curves (Gini).
6. **Interpretation** with GLM relativities and SHAP.

## Results (test set)

| Model | Poisson deviance | Deviance explained | Gini |
|---|---|---|---|
| Baseline (portfolio average) | 0.6274 | 0.0% | 0 |
| Poisson GLM | 0.5944 | 5.3% | 0.285 |
| **LightGBM** | **0.5762** | **8.2%** | **0.335** |

## Key findings
- **Bonus-malus** is the strongest risk factor: drivers in the malus zone (130+) claim about 4.4 times more often than average.
- **Young drivers' risk is largely explained by bonus-malus:** they have not had time to earn a bonus (average 95 at ages 18–20 vs about 53 over 50), so once bonus-malus is in the model, the separate effect of age is much smaller.
- **New vehicles of brand B12** have a claim frequency of about 0.45, roughly 4.5 times the average, while older B12 vehicles are slightly below average. LightGBM learned this interaction on its own, whereas the GLM needed it added by hand.
- **LightGBM beats the GLM** in both accuracy and risk ranking (Gini 0.335 vs 0.285), while SHAP keeps it explainable.

## Limitations and next steps
- Some policyholders appear with split policies, so a grouped train/test split would give a stricter evaluation.
- The cause of the B12 anomaly would need to be checked with the business.
- Adding a severity model would complete the pure premium.
- Hyperparameter tuning could reduce LightGBM's overfitting.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/pricing_model.ipynb
```