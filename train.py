import pandas as pd
import joblib
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import os

df = pd.read_csv('datasets/merged_dataset.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

X = df.drop('28-day Compressive Strength (N/mm2) M30', axis=1)
y = df['28-day Compressive Strength (N/mm2) M30']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model = Lasso(alpha=0.1)
final_model.fit(X_scaled, y)

os.makedirs('assests/models', exist_ok=True)

joblib.dump(final_model, 'assests/models/lasso_model.pkl')
joblib.dump(scaler, 'assests/models/scaler.pkl')

print("Success! Models have been successfully retrained and saved locally.")