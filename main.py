
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

input_path = r"C:\Users\polym\Desktop\Testing123\bmw.csv"
output_dir = r"C:\Users\polym\Desktop\Testing123\outputs"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)

df = df.drop_duplicates()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

df_encoded = pd.get_dummies(df, drop_first=True)

X_vif = df_encoded.drop(columns=['price'])

X_vif = X_vif.select_dtypes(include=[np.number])

vif_df = pd.DataFrame()
vif_df['feature'] = X_vif.columns
vif_df['VIF'] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

vif_df.to_csv(os.path.join(output_dir, "vif_report.csv"), index=False)

X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
results_df.to_csv(os.path.join(output_dir, "model_metrics.csv"), index=False)

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="year", y="price", alpha=0.5)
plt.title("Зависимость цены от года выпуска")
plt.savefig(os.path.join(output_dir, "price_vs_year.png"))
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="mileage", y="price", alpha=0.5)
plt.title("Зависимость цены от пробега")
plt.savefig(os.path.join(output_dir, "price_vs_mileage.png"))
plt.close()

df_encoded.to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)

print("Готово! Все файлы сохранены в:", output_dir)
