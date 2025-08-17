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

# Function to predict next best action for a given customer ID
def predict_next_best_action(customer_id, df=df, model=m):
    shopper_type = df.loc[df['CST_ID'] == customer_id, 'SHOPPER'].iloc[0]
    df_shopper = df[df['SHOPPER'] == shopper_type]

    data = prepare_data(df_shopper)

    if customer_id not in data['CST_ID'].values:
        return f"Customer ID {customer_id} not found for shopper type {shopper_type}"

    customer_data = data[data['CST_ID'] == customer_id].drop(columns=['CST_ID', 'NEXT_PRODUCT_TYPE'])

    customer_data['PRODUCT_TYPE'] = label_encoder_product_type.transform(customer_data['PRODUCT_TYPE'])
    customer_data['PRODUCT_BRAND'] = label_encoder_product_brand.transform(customer_data['PRODUCT_BRAND'])
    customer_data['LAST_PRODUCT_TYPE'] = label_encoder_last_product_type.transform(customer_data['LAST_PRODUCT_TYPE'])

    next_product_type_encoded = model.predict(customer_data)[0]

    next_product_type = label_encoder_next_product_type.inverse_transform([next_product_type_encoded])[0]

    return next_product_type

with col1:
    st.subheader("Predict from Customer ID")
    unique_id = sorted(df['CST_ID'].unique())
    customer_id = st.selectbox("Customer ID", unique_id)
    next_best_action = predict_next_best_action(customer_id, nba_df)
    st.write(f"The next best action for customer ID {customer_id} ({shopper_type} shopper) is to recommend the product type: {next_best_action}")

with col2:
    
    st.subheader("Predict from dataset row")
    
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



st.markdown("---")
st.subheader("Notes / Troubleshooting")
