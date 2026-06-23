# Pune Smart Towing Intelligence System

A Streamlit application for predicting vehicle towing risk and finding the
assigned towing yard across Pune's traffic divisions.

## Files
- `app.py` — the Streamlit application
- `Pune_With_Real_Features_v5.csv` — dataset (must stay in the same folder as `app.py`)
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Modules
1. **Dashboard** — project overview, KPI stats, quick insights
2. **Risk Predictor** — predicts towing risk for a road/area using a RandomForest model
3. **Yard Finder** — looks up the assigned towing yard for a road/division
4. **Analytics** — high-risk areas, towing distribution, yard distribution, area-wise insights
5. **About Project** — description, tech stack, dataset summary

## Notes
- The ML feature engineering, label encoding, and `RandomForestClassifier`
  hyperparameters are unchanged from the original research notebook
  (`real_world_data.ipynb`).
- "Vehicle Type" is collected in the Risk Predictor for context/display only —
  the dataset has no vehicle-type column, so it does not influence the model,
  per the requirement to keep the existing ML logic unchanged.
