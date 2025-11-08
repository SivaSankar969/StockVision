import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="📊 Stock Price Prediction Dashboard", layout="centered")
st.title("📊 Stock Price Prediction Dashboard")
st.markdown("Predict future stock prices with Machine Learning")
def detect_companies():
    companies = {}
    model_files = [f for f in os.listdir("models") if f.endswith("_model.pkl")]
    for model_file in model_files:
        company_name = model_file.replace("_model.pkl", "")
        companies[company_name] = {
            "model": f"models/{company_name}_model.pkl",
            "scaler": f"models/{company_name}_scaler.pkl",
            "data": f"data/processed/{company_name}_processed.csv"
        }
    return companies

companies = detect_companies()

if not companies:
    st.error("❌ No trained models found in the 'models/' folder. Please train at least one model first.")
    st.stop()

selected_company = st.selectbox("🏢 Select a company", list(companies.keys()))
paths = companies[selected_company]
for key, path in paths.items():
    if not os.path.exists(path):
        st.error(f"❌ Missing {key} file: {path}. Please train the model first.")
        st.stop()

model = joblib.load(paths["model"])
scaler = joblib.load(paths["scaler"])
df = pd.read_csv(paths["data"])
st.subheader("📅 Prediction Options")
option = st.radio("What do you want to predict?", ["Today", "Tomorrow", "Custom Range"])

if option == "Custom Range":
    start_date = st.date_input("Start Date", datetime.today())
    end_date = st.date_input("End Date", datetime.today() + timedelta(days=7))
    future_days = (end_date - start_date).days
elif option == "Tomorrow":
    start_date = datetime.today()
    future_days = 1
else:
    start_date = datetime.today()
    future_days = 0

if st.button("🔮 Predict Future Prices"):
    st.info(f"Predicting {future_days} day(s) ahead for {selected_company}...")

    last_row = df.iloc[-1].copy()
    future_predictions = []
    future_dates = [start_date + timedelta(days=i) for i in range(1, future_days + 1)]

    for _ in range(future_days):
        X_input = last_row[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'MA_5', 'MA_20']].to_frame().T
        X_scaled = scaler.transform(X_input)
        pred_close = model.predict(X_scaled)[0]
        future_predictions.append(pred_close)

        last_row['Prev_Close'] = pred_close
        last_row['MA_5'] = (last_row['MA_5'] * 4 + pred_close) / 5
        last_row['MA_20'] = (last_row['MA_20'] * 19 + pred_close) / 20

    result_df = pd.DataFrame({"Date": future_dates, "Predicted Close (₹)": future_predictions})
    st.success("✅ Future prediction completed!")
    st.dataframe(result_df)

    plt.figure(figsize=(10, 5))
    plt.plot(result_df["Date"], result_df["Predicted Close (₹)"], marker='o', color='blue', label='Predicted Price')
    plt.title(f"{selected_company} Future Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Predicted Close Price (₹)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    future_graph_path = f"outputs/{selected_company}_future_prediction.png"
    plt.savefig(future_graph_path)
    st.pyplot(plt)
    st.success(f"📈 Future graph saved at: {future_graph_path}")

    st.subheader("📊 Historical Price Trend")
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], color='green', label='Historical Close')
    plt.title(f"{selected_company} Historical Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Close Price (₹)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    history_graph_path = f"outputs/{selected_company}_historical_trend.png"
    plt.savefig(history_graph_path)
    st.pyplot(plt)
    st.success(f"📉 Historical trend graph saved at: {history_graph_path}")
