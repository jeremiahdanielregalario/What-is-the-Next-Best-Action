import streamlit as st
import os
import pandas as pd
import joblib
import pickle
import json
import traceback
import numpy as np
from typing import Any, Dict, List, Optional, Union

st.set_page_config(page_title="Next Best Action", layout="wide")

@st.cache_data
def load_df():
    p = "nba.parquet"
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        # return None and the error so UI can show it
        return None, p, str(e)
    return df, p, None

@st.cache_data
def load_model():
    p = "model.pkl"
    try:
        m = joblib.load(p)
    except Exception:
        try:
            with open(p, "rb") as f:
                m = pickle.load(f)
        except Exception as e:
            return None, p, str(e)
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

# Function to predict next best action for a given customer ID
def predict_next_best_action(
    customer_id,
    df: pd.DataFrame = df,
    model = model,
    id_col: str = "CST_ID",
    action_col: str = "PRODUCT_TYPE",
    feature_cols: Optional[List[str]] = None,
    actions: Optional[List[Any]] = None,
    shopper_type_col: Optional[str] = "SHOPPER",
    return_scores: bool = False
):
    # 1. find customer row
    matches = df[df[id_col] == customer_id]
    if matches.shape[0] == 0:
        raise ValueError(f"No row found in dataframe for {id_col} == {customer_id!r}")
    customer_row = matches.iloc[0]

    # Get shopper_type (if available)
    shopper_type = None
    if shopper_type_col and shopper_type_col in df.columns:
        shopper_type = customer_row.get(shopper_type_col)

    # 2. determine candidate actions
    if actions is None:
        if action_col in df.columns:
            actions = pd.Series(df[action_col].unique()).tolist()
        else:
            raise ValueError("No `actions` provided and `action_col` not present in df to derive candidates.")

    # 3. determine feature columns
    if feature_cols is None:
        exclude = {id_col, action_col}
        if shopper_type_col:
            exclude.add(shopper_type_col)
        feature_cols = [c for c in df.columns if c not in exclude]

    # Determine expected feature columns from model if available
    if model is not None and hasattr(model, "feature_names_in_"):
        feature_cols_expected = list(model.feature_names_in_)
    else:
        feature_cols_expected = list(feature_cols)

    # Base feature vector for the customer
    base_features = customer_row.copy()

    # 4. Build candidate rows (one per action)
    candidate_rows = []
    candidate_labels = []
    for act in actions:
        row = base_features.copy()

        # If the action was a feature during training, set it for the candidate row.
        if action_col in feature_cols_expected:
            row[action_col] = act

        candidate_rows.append(row)
        candidate_labels.append(act)

    X_candidates = pd.DataFrame(candidate_rows)

    # Ensure shopper_type_col is present if model expects it but it's missing in candidate rows:
    if shopper_type_col and shopper_type_col in feature_cols_expected and shopper_type_col not in X_candidates.columns:
        X_candidates[shopper_type_col] = shopper_type

    # Align columns to model expectation and fill missing with NaN
    for c in feature_cols_expected:
        if c not in X_candidates.columns:
            X_candidates[c] = np.nan
    # Select columns in expected order
    X_candidates = X_candidates[feature_cols_expected]

    # 5. Scoring
    scores: Dict[Any, float] = {}

    # Case A: classifier with predict_proba and classes_ (we interpret per-candidate probability for candidate label)
    if model is not None and hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        try:
            proba = model.predict_proba(X_candidates)  # shape (n_samples, n_classes)
            classes = list(model.classes_)
            for i, act in enumerate(candidate_labels):
                if act in classes:
                    cls_idx = classes.index(act)
                    scores[act] = float(proba[i][cls_idx])
                else:
                    # If candidate label not present in classes, use the model's max probability as fallback
                    scores[act] = float(np.max(proba[i]))
        except Exception:
            scores = {}

    # Case B: numeric predictions or other predict behavior
    if not scores:
        if model is None:
            raise RuntimeError("No model available to score candidates.")
        try:
            preds = model.predict(X_candidates)
            if np.issubdtype(np.array(preds).dtype, np.number):
                # higher predicted numeric value considered better
                for i, act in enumerate(candidate_labels):
                    scores[act] = float(preds[i])
            else:
                # Non-numeric labels: score by whether predicted label equals the candidate label
                for i, act in enumerate(candidate_labels):
                    scores[act] = 1.0 if preds[i] == act else 0.0
        except Exception as e:
            raise RuntimeError("Model prediction failed for candidate simulations: " + str(e))

    if len(scores) == 0:
        raise RuntimeError("Could not compute scores for any action candidates.")

    # Select best action (highest score). Break ties by first occurrence.
    best_action = max(scores.items(), key=lambda kv: kv[1])[0]

    if return_scores:
        return best_action, scores, shopper_type
    return best_action, shopper_type


with col1:
    st.subheader("Predict from Customer ID")
    if df is None:
        st.info("No dataframe loaded.")
    else:
        # defensively handle missing column
        if 'CST_ID' not in df.columns:
            st.error("Column 'CST_ID' not found in data.")
        else:
            try:
                unique_id = sorted(list(df['CST_ID'].dropna().unique()))
            except Exception:
                unique_id = list(df['CST_ID'].dropna().unique())
            customer_id = st.selectbox("Customer ID", unique_id)
            try:
                best_action, shopper_type = predict_next_best_action(customer_id)
                st.write(f"The next best action for customer ID {customer_id} ({shopper_type} shopper) is to recommend the product type: **{best_action}**")
            except Exception as e:
                st.exception(e)


with col2:
    st.subheader("Predict from dataset row")

    try:
        idx_options = list(map(str, list(df.index[:500])))
    except Exception:
        idx_options = [str(i) for i in range(min(500, len(df)))]
    selected_idx = st.selectbox("Select row index", idx_options)
    if st.button("Predict selected row"):
        # try to resolve selected_idx to an actual index label
        row = None
        # attempt to coerce to same type as df.index[0]
        try:
            idx0 = df.index[0]
            if isinstance(idx0, str):
                idx_val = selected_idx
            else:
                # try convert to numeric index type (int, numpy.int64, etc.)
                try:
                    idx_val = type(idx0)(selected_idx)
                except Exception:
                    # fallback to int conversion
                    idx_val = int(selected_idx)
            row = df.loc[idx_val]
        except Exception:
            try:
                # fallback: positional
                row = df.iloc[int(selected_idx)]
            except Exception as e:
                st.error(f"Could not find row: {e}")
                row = None

        if row is not None:
            # prepare features
            try:
                if model is not None and hasattr(model, "feature_names_in_"):
                    cols = list(model.feature_names_in_)
                    # if columns missing from row, fill with NaN
                    vals = []
                    for c in cols:
                        vals.append(row[c] if c in row.index else np.nan)
                    X = pd.DataFrame([vals], columns=cols)
                else:
                    # use the row's own columns
                    X = pd.DataFrame([row.values], columns=list(row.index))
                pred = model.predict(X)
                # present prediction (force tolist for nice display)
                try:
                    st.success(f"Prediction: {pd.Series(pred).tolist()}")
                except Exception:
                    st.success(f"Prediction: {pred}")
                if hasattr(model, "predict_proba"):
                    st.write("Probabilities:", model.predict_proba(X).tolist())
            except Exception as e:
                st.exception(e)


st.markdown("---")
st.subheader("Notes / Troubleshooting")
