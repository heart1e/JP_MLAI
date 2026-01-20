0. Global Configurations — consolidation/cleanup pass

1.1 Column Aliases and Identity Checks — add alias
1.2 Load AAPL Data with Caching — sanity check

2.1 Feature Engineering — feature iteration
2.1.1 Derived Drivers (DSO/DPO/DIH, Margins, Growth, Logs) — deeper features
2.2 Dataset Assembly — minimal dataset
2.3 Prev-state Matrix for Algebraic Layer — for algebraic layer 
2.4 Scaling (z-score) for Stability — add scaler
2.5 Train/Val Split on Scaled Data — fix evaluation 


3.1 TF Model with Algebraic Generator + Earnings Head — baseline model
3.2 Train/Evaluate (MAE on BS + Earnings) — quick run 