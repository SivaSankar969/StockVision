import pandas as pd
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE_NAME = input("enter the file path which is in processed: ")

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(f"❌ File '{FILE_NAME}' not found! Please run preprocess first.")

print(f"\n📂 Loading processed data: {FILE_NAME}")
df = pd.read_csv(FILE_NAME)

required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Prev_Close', 'MA_5', 'MA_20']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"❌ Missing required column: '{col}' in the processed CSV!")

X = df[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'MA_5', 'MA_20']]
y = df['Close']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = (1 - (mae / np.mean(y_test))) * 100
accuracy = max(0, min(100, accuracy))

print("\n📊 Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"✅ Model Accuracy: {accuracy:.2f}%")

os.makedirs("models", exist_ok=True)
company_name = os.path.splitext(os.path.basename(FILE_NAME))[0].replace('_processed', '')
model_path = f"models/{company_name}_model.pkl"
scaler_path = f"models/{company_name}_scaler.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✅ Model training complete and saved successfully!")
print(f"📁 Model Path: {model_path}")
print(f"📁 Scaler Path: {scaler_path}")

os.makedirs("results", exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", color='blue')
plt.plot(y_pred, label="Predicted", color='red', linestyle='dashed')
plt.title(f"{company_name.upper()} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)

plot_path = f"results/{company_name}_prediction_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"📈 Prediction graph saved at: {plot_path}")
