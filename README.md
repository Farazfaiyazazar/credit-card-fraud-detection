# Credit Card Fraud Detection

Detecting fraudulent transactions in a real, highly imbalanced dataset (492 fraud cases out of 285k transactions, 0.17%).

## Dataset
European credit card transactions from September 2013 (get it from Kaggle/ULB). Features are PCA transformed for privacy (V1-V28), plus `Time` and `Amount`.

## Process
1. **EDA** — explored class imbalance, transaction amount patterns, time-of-day patterns, and correlation of features with fraud
2. **Key findings:**
   - Fraud is disproportionately concentrated at low transaction amounts, with a secondary spike around $100
   - Fraud spikes sharply around 2-3 AM, likely exploiting low- traffic hours
   - Top correlated features (V17, V14, V12, V10) show fraud as scattered outliers rather than a separate cluster — no single feature cleanly separates classes
3. **Modeling** — compared 6 models to handle the extreme class imbalance:

| Model | Precision | Recall |
|---|---|---|
| Logistic Regression (baseline) | 0.88 | 0.62 |
| Logistic Regression (scaled) | 0.85 | 0.59 |
| Logistic Regression (class-weighted) | 0.06 | 0.87 |
| Logistic Regression + SMOTE | 0.05 | 0.87 |
| Random Forest (class-weighted) | 0.96 | 0.72 |
| **XGBoost (class-weighted)** | **0.91** | **0.78** |

4. **Result:** XGBoost gave the best overall balance of catching fraud (78% recall) while keeping false alarms low (91% precision). Linear models hit a hard precision/recall tradeoff regardless of imbalance-handling technique; tree-based models handled the non-linear fraud patterns much better.

## Tools
Python, pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn

## Notes
This is a learning project so the goal was to practice EDA, understand class imbalance techniques, and compare models properly rather than just picking one and stopping.