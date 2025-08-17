import streamlit as st
import os
import pandas as pd
import joblib
import pickle
import json
import traceback
import numpy as np

st.set_page_config(page_title="Next Best Action", layout="wide")

@st.cache_data
def load_df():
    last_err = None
    p = "nba.parquet"
    return pd.read_parquet(p), p, None

@st.cache_data
def load_model():
    last_err = None
    p = "model.pkl"
    try:
        m = joblib.load(p)
    except Exception:
        with open(p, "rb") as f:
            m = pickle.load(f)
    return m, p, None

df, df_path, df_error = load_df()
model, model_path, model_error = load_model()

st.title("Next Best Action")
st.markdown(f"**Data:** `{df_path or 'not found'}`  &nbsp;  |  &nbsp; **Model:** `{model_path or 'not found'}`")

if df is None:
    st.error("Dataframe not found. Place nba.parquet in /mnt/data or project root.")
    if df_error:
        st.code(df_error)
else:
    st.subheader("Data sample")
    st.dataframe(df.head(50))

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Predict from dataset row")
    # Create readable index options
    try:
        idx_options = list(map(str, list(df.index[:500])))
    except Exception:
        idx_options = [str(i) for i in range(min(500, len(df)))]
    selected_idx = st.selectbox("Select row index", idx_options)
    if st.button("Predict selected row"):
        try:
            # locate row
            try:
                # try same dtype index
                row = df.loc[type(df.index[0])(selected_idx)]
            except Exception:
                # fallback: treat as integer position
                row = df.loc[selected_idx]
        except Exception:
            try:
                row = df.iloc[int(selected_idx)]
            except Exception as e:
                st.error(f"Could not find row: {e}")
                row = None
        if row is not None:
            # prepare features
            if hasattr(model, "feature_names_in_"):
                cols = list(model.feature_names_in_)
                X = pd.DataFrame([row[cols].values], columns=cols)
            else:
                X = pd.DataFrame([row.values], columns=list(row.index))
            try:
                pred = model.predict(X)
                st.success(f"Prediction: {pred.tolist()}")
                if hasattr(model, "predict_proba"):
                    st.write("Probabilities:", model.predict_proba(X).tolist())
            except Exception as e:
                st.exception(e)

with col2:
    st.subheader("Predict from JSON")
    json_text = st.text_area("Paste a JSON dict or list of dicts (features)", height=200, value="{}")
    if st.button("Predict JSON"):
        try:
            payload = json.loads(json_text)
            if isinstance(payload, dict):
                X = pd.DataFrame([payload])
            elif isinstance(payload, list):
                X = pd.DataFrame(payload)
            else:
                st.error("Unsupported JSON format: must be dict or list of dicts")
                X = None
            if X is not None:
                if hasattr(model, "feature_names_in_"):
                    needed = list(model.feature_names_in_)
                    for c in needed:
                        if c not in X.columns:
                            X[c] = np.nan
                    X = X[needed]
                pred = model.predict(X)
                st.success(f"Predictions: {pred.tolist()}")
                if hasattr(model, "predict_proba"):
                    st.write("Probabilities:", model.predict_proba(X).tolist())
        except Exception as e:
            st.exception(e)

st.markdown("---")
st.subheader("Notes / Troubleshooting")
if model is None:
    st.warning("Model not loaded. Place model.pkl in the project root or /mnt/data.")
    if model_error:
        st.code(model_error)