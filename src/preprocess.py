import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="Path to stock CSV file")
args = parser.parse_args()

file_name = args.file or input("Enter the stock CSV filename (e.g., reliance_stock.csv): ").strip()

if not os.path.exists(file_name):
    raise FileNotFoundError(f"❌ File '{file_name}' not found!")

print("\n📂 Loading file:", file_name)
df = pd.read_csv(file_name)

print("\n🧾 Columns detected in file:")
print(df.columns.tolist())

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
elif df.columns[0].lower().startswith('unnamed'):
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
else:
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in numeric_cols:
    if col not in df.columns:
        print(f"⚠️ Missing column: {col}, trying to auto-detect numeric columns.")
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

df.dropna(inplace=True)

df = df.sort_values('Date')

df['Prev_Close'] = df['Close'].shift(1)
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()
df.dropna(inplace=True)

os.makedirs("data/processed", exist_ok=True)
company_name = os.path.splitext(os.path.basename(file_name))[0]
processed_path = f"data/processed/{company_name}_processed.csv"
df.to_csv(processed_path, index=False)

print(f"\n✅ Preprocessing complete for {company_name.upper()}!")
print(f"📁 Saved processed data as: {processed_path}")
print("\n🧩 Preview:")
print(df.head())
