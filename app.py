import streamlit as st
import os
import pandas as pd
import joblib
import pickle
import traceback
import numpy as np
from typing import Any, Dict, List, Optional

st.set_page_config(page_title="Next Best Action", layout="wide")

@st.cache_data
def load_df() -> (Optional[pd.DataFrame], str, Optional[str]):
    p = "nba.parquet"
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        return None, p, str(e)
    return df, p, None

@st.cache_data
def load_model() -> (Any, str, Optional[str]):
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
    st.error("Dataframe not found. Place nba.parquet in project root.")
    if df_error:
        st.code(df_error)
else:
    st.subheader("Data sample")
    st.dataframe(df.head(50))

# # Show quick model diagnostics to help debugging
# st.subheader("Model diagnostics (helpful for debugging)")
# try:
#     st.write("Model type:", type(model))
#     st.write("Has predict_proba?:", hasattr(model, "predict_proba"))
#     st.write("Has predict?:", hasattr(model, "predict"))
#     st.write("Has classes_?:", hasattr(model, "classes_"))
#     st.write("Pipeline named_steps (if pipeline):", getattr(model, "named_steps", None))
#     try:
#         st.write("feature_names_in_:", list(model.feature_names_in_))
#     except Exception:
#         st.write("feature_names_in_: (not present)")
# except Exception:
#     st.write("Could not fetch diagnostics for model.")

col1, col2 = st.columns([2, 1])

def _coerce_candidates(X: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Try to coerce candidate dataframe columns to types similar to df_train.
    - If df_train column is numeric, attempt numeric conversion.
    - If df_train column is categorical/object, map categories -> numeric codes using categories present in df_train.
    - For unknown columns, map object columns to categorical codes.
    Returns a copy of X converted to numeric-ish values where possible.
    """
    Xc = X.copy()
    for c in Xc.columns:
        try:
            if c in df_train.columns:
                train_dtype = df_train[c].dtype
                if pd.api.types.is_numeric_dtype(train_dtype):
                    # try numeric conversion (coerce errors to NaN)
                    Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
                elif pd.api.types.is_categorical_dtype(train_dtype) or train_dtype == object:
                    # map using categories present in training df
                    cats = pd.Categorical(df_train[c].astype("category")).categories
                    mapping = {cat: i for i, cat in enumerate(cats)}
                    # map unknowns to -1, then cast to float
                    Xc[c] = Xc[c].map(mapping).fillna(-1).astype(float)
                else:
                    # fallback: attempt numeric conversion
                    Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
            else:
                # column not in training df: if object-like, convert to categorical codes
                if Xc[c].dtype == object or pd.api.types.is_categorical_dtype(Xc[c].dtype):
                    Xc[c] = pd.Categorical(Xc[c]).codes.astype(float)
                else:
                    Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
        except Exception:
            # last-resort: convert to numeric, coerce errors
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
    return Xc

# Main function
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
    if df is None:
        raise RuntimeError("No dataframe loaded.")
    # 1. find customer row
    matches = df[df[id_col] == customer_id]
    if matches.shape[0] == 0:
        raise ValueError(f"No row found in dataframe for {id_col} == {customer_id!r}")
    customer_row = matches.iloc[0]

    # shopper_type if present
    shopper_type = None
    if shopper_type_col and shopper_type_col in df.columns:
        shopper_type = customer_row.get(shopper_type_col)

    # 2. candidate actions
    if actions is None:
        if action_col in df.columns:
            actions = pd.Series(df[action_col].unique()).tolist()
        else:
            raise ValueError("No `actions` provided and `action_col` not present in df to derive candidates.")

    # 3. feature columns guess
    if feature_cols is None:
        exclude = {id_col, action_col}
        if shopper_type_col:
            exclude.add(shopper_type_col)
        feature_cols = [c for c in df.columns if c not in exclude]

    # expected features from model if present
    if model is not None and hasattr(model, "feature_names_in_"):
        feature_cols_expected = list(model.feature_names_in_)
    else:
        feature_cols_expected = list(feature_cols)

    # base features
    base_features = customer_row.copy()

    # build candidate rows
    candidate_rows = []
    candidate_labels = []
    for act in actions:
        row = base_features.copy()
        if action_col in feature_cols_expected:
            row[action_col] = act
        candidate_rows.append(row)
        candidate_labels.append(act)

    X_candidates = pd.DataFrame(candidate_rows)

    # ensure shopper type presence if expected
    if shopper_type_col and shopper_type_col in feature_cols_expected and shopper_type_col not in X_candidates.columns:
        X_candidates[shopper_type_col] = shopper_type

    # align to expected feature columns; add missing ones with NaN
    for c in feature_cols_expected:
        if c not in X_candidates.columns:
            X_candidates[c] = np.nan
    # choose column order
    if len(feature_cols_expected) > 0:
        X_candidates = X_candidates[feature_cols_expected]
    else:
        # if model didn't provide feature names, keep X_candidates as-is
        X_candidates = X_candidates

    # Scoring containers
    scores: Dict[Any, float] = {}

    # Try direct prediction first (works if model is a full Pipeline that accepts raw df)
    proba = None
    preds = None
    prediction_error_first = None

    try:
        if model is None:
            raise RuntimeError("No model loaded.")
        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            proba = model.predict_proba(X_candidates)
        else:
            preds = model.predict(X_candidates)
    except Exception as e_first:
        prediction_error_first = e_first
        # Attempt to coerce candidate dtypes to match df and retry
        X_safe = _coerce_candidates(X_candidates, df)
        try:
            if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
                proba = model.predict_proba(X_safe)
                X_candidates = X_safe
            else:
                preds = model.predict(X_safe)
                X_candidates = X_safe
        except Exception as e_second:
            # find columns that still have non-numeric samples
            non_numeric = []
            for col in X_safe.columns:
                sample_vals = X_safe[col].dropna().iloc[:10] if X_safe[col].dropna().shape[0] > 0 else []
                ok = True
                for v in sample_vals:
                    try:
                        float(v)
                    except Exception:
                        ok = False
                        break
                if not ok:
                    non_numeric.append(col)
            raise RuntimeError(
                "Model prediction failed after attempting dtype coercion. "
                f"Original error: {e_first}; Retry error: {e_second}. "
                f"Columns with remaining non-numeric sample values: {non_numeric}"
            )

    # If we have probabilities from a classifier
    if proba is not None and hasattr(model, "classes_"):
        classes = list(model.classes_)
        for i, act in enumerate(candidate_labels):
            if act in classes:
                cls_idx = classes.index(act)
                scores[act] = float(proba[i][cls_idx])
            else:
                scores[act] = float(np.max(proba[i]))
    else:
        # use preds (numeric or labels)
        if preds is None:
            raise RuntimeError("Prediction did not produce probabilities or predictions.")
        # if predictions numeric values, use them directly; else compare equality
        if np.issubdtype(np.array(preds).dtype, np.number):
            for i, act in enumerate(candidate_labels):
                scores[act] = float(preds[i])
        else:
            for i, act in enumerate(candidate_labels):
                scores[act] = 1.0 if preds[i] == act else 0.0

    if len(scores) == 0:
        raise RuntimeError("Could not compute scores for any action candidates.")

    best_action = max(scores.items(), key=lambda kv: kv[1])[0]

    if return_scores:
        return best_action, scores, shopper_type
    return best_action, shopper_type

# UI: Customer ID prediction
with col1:
    st.subheader("Predict from Customer ID")
    if df is None:
        st.info("No dataframe loaded.")
    else:
        if 'CST_ID' not in df.columns:
            st.error("Column 'CST_ID' not found in data.")
        else:
            unique_id = sorted(list(df['CST_ID'].dropna().unique()))
            customer_id = st.selectbox("Customer ID", unique_id)
            if st.button("Predict for selected customer"):
                try:
                    best_action, shopper_type = predict_next_best_action(customer_id)
                    st.success(f"The next best action for customer ID {customer_id} ({shopper_type}) is: **{best_action}**")
                except Exception as e:
                    st.exception(e)

# UI: Row-based prediction
with col2:
    st.subheader("Predict from dataset row")
    try:
        idx_options = list(map(str, list(df.index[:500])))
    except Exception:
        idx_options = [str(i) for i in range(min(500, len(df)))]
    selected_idx = st.selectbox("Select row index", idx_options)
    if st.button("Predict selected row"):
        row = None
        try:
            idx0 = df.index[0]
            if isinstance(idx0, str):
                idx_val = selected_idx
            else:
                try:
                    idx_val = type(idx0)(selected_idx)
                except Exception:
                    idx_val = int(selected_idx)
            row = df.loc[idx_val]
        except Exception:
            try:
                row = df.iloc[int(selected_idx)]
            except Exception as e:
                st.error(f"Could not find row: {e}")
                row = None

        if row is not None:
            try:
                if model is not None and hasattr(model, "feature_names_in_"):
                    cols = list(model.feature_names_in_)
                    vals = []
                    for c in cols:
                        vals.append(row[c] if c in row.index else np.nan)
                    X = pd.DataFrame([vals], columns=cols)
                else:
                    X = pd.DataFrame([row.values], columns=list(row.index))
                pred = model.predict(X)
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
st.write("""
         - Short-term: this app will attempt to coerce candidate rows to numeric codes using the training `nba.parquet`, but that mapping may differ from your exact encoder used at training time.
""")
