# What is the Next Best Action?
Next Best Action Ad Hoc project from BPI Data Science Internship 2024 Project

Check out the online streamlit app [here](https://what-is-the-next-best-action.streamlit.app/)

A lightweight Replit-ready app for making predictions with your pre-trained NBA model.

This repository contains a single-file Flask app (and an optional Streamlit example) that loads:

- `model.pkl` — your trained scikit-learn model (place in project root or `/mnt/data`)
- `nba.parquet` — the dataset used for samples and UI (place at `/mnt/data/nba.parquet` or project root)

The app provides a minimal web UI and a programmatic REST API so you can both demo interactively and integrate the predictor into other tools.

---

## Features

- Loads model from `model.pkl` (supports `joblib` or `pickle` files).
- Loads dataset from `nba.parquet` for quick sample/pick-a-row predictions.
- Web UI for browsing sample rows and sending predictions.
- REST endpoints for programmatic predictions:
  - `POST /api/predict` — JSON `{ "features": {...} }` or directly a feature dict/list
  - `GET  /api/sample?n=5` — get sample rows from the dataset
  - `POST /predict_row` — (UI form) predict using a selected dataset row
  - `POST /predict_json` — (UI form) predict using pasted JSON
- Graceful fallbacks and helpful error messages if files are missing or loading fails.

---

## Repository files

- `replit-nba-predictor-app.py` — single-file Flask app (main application)
- `streamlit_nba_predictor.py` — optional Streamlit demo (example, not required)
- `requirements.txt` — dependency list (populate with the appropriate block shown below)
- `model.pkl` — **your** trained model (not checked in)
- `nba.parquet` — **your** dataset file (not checked in)

---

## Recommended `requirements.txt`

For the Streamlit demo:

```
streamlit
pandas
numpy
pyarrow
scikit-learn
joblib
```

Optional extras:
- `fastparquet` (alternative parquet engine)
- `matplotlib` (plots)
- `gunicorn` (production WSGI)

---

## Setup & Run (Replit)

1. Add `model.pkl` and `nba.parquet` to your Replit workspace. You can put the dataset in `/mnt/data/nba.parquet` — the app already checks there.
2. Add the chosen `requirements.txt` to the project root.
3. Set the Replit run command to:

For Flask (single-file app):

```
python replit-nba-predictor-app.py
```

For Streamlit (optional):

```
streamlit run streamlit_nba_predictor.py --server.port $PORT --server.headless true
```

Replit will automatically expose the web view — the Flask app listens on port `3000` by default (or `$PORT` if provided).

---

## Setup & Run (Locally)

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate  # Windows (PowerShell)

pip install -r requirements.txt
```

2. Ensure `model.pkl` and `nba.parquet` are in the project root (or update paths in the script).

3. Run the Flask app:

```bash
python replit-nba-predictor-app.py
```

Open http://localhost:3000 in your browser.

If using Streamlit:

```bash
streamlit run streamlit_nba_predictor.py
```

Open http://localhost:8501 (or the printed local URL).

---

## API Usage Examples

**Predict from JSON (curl)**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 12.3, "feature2": 4}}' \
  http://localhost:3000/api/predict
```

**Predict a list of feature dicts**

```bash
curl -X POST -H "Content-Type: application/json" -d '[{"f1":1, "f2":2}, {"f1":3, "f2":4}]' http://localhost:3000/api/predict
```

**Get sample rows**

```bash
curl http://localhost:3000/api/sample?n=10
```

---

## How the predictor prepares inputs

- If your model object has `feature_names_in_` (typical for scikit-learn transformers/estimators), the app will attempt to align JSON or DataFrame columns to that expectation and fill missing columns with `NaN`.
- When you pick a row from the dataset UI, the app converts that row to a single-row DataFrame before calling `model.predict`.
- If your model exposes `predict_proba`, the API will return probability scores in the response.

---

## Troubleshooting

- **Dataframe not found / parquet import error**: verify `nba.parquet` exists in `/mnt/data` or in project root. If Parquet reading fails, try converting to CSV (`pandas.read_parquet` vs `pandas.read_csv`) or install `fastparquet`.
- **Model load errors**: ensure `model.pkl` was saved with `joblib.dump()` or `pickle.dump()`; the app tries both `joblib.load()` and `pickle.load()`.
- **Port conflicts on Replit**: the Flask app uses port `3000` by default; Replit exposes `$PORT` — you can set the `PORT` env var in Replit or modify `APP_PORT` in the script.

If you hit an error, the app prints tracebacks to help debugging and surfaces them in the UI.

---

## Author

- [Jeremiah Daniel A. Regalario](https://github.com/jeremiahdanielregalario/)


