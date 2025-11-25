
# подключаем и импортируем необходимые библиотеки
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

# указываем путь датасета и путь для сохранения результатов
input_path = r"C:\Users\polym\Desktop\Testing123\bmw.csv"
output_dir = r"C:\Users\polym\Desktop\Testing123\outputs"

# если папки нет, то она создастся автоматически
os.makedirs(output_dir, exist_ok=True)

# загружаем сам датасет
df = pd.read_csv(input_path)

# убираем дубликаты
df = df.drop_duplicates()

# заполняем пропуски самыми популярными значениями(модой) в категориальных признаках
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# в числовых признаках заполняем пропуски медианой
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# переводим категориальные переменные в набор dummy-признаков
df_encoded = pd.get_dummies(df, drop_first=True)

# берём все признаки, кроме целевой переменной price
X_vif = df_encoded.drop(columns=['price'])

# фильтруме столбцы чтобы брать только числовые данные(vif работает только с числовыми)
X_vif = X_vif.select_dtypes(include=[np.number])

# формируем vif таблицу
vif_df = pd.DataFrame()
vif_df['feature'] = X_vif.columns
vif_df['VIF'] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

# сохраняем отчёт о мультиколлинеарности
vif_df.to_csv(os.path.join(output_dir, "vif_report.csv"), index=False)

# разделяем датасет на признками и целевую переменную
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# 20% пойдут на обучение, а остальные на тест
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
}

results = []

# тут 3 разные модели с циклом их обучения
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
results_df.to_csv(os.path.join(output_dir, "model_metrics.csv"), index=False)

# сохраняем график зависимости цены от года выпуска
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="year", y="price", alpha=0.5)
plt.title("Зависимость цены от года выпуска")
plt.savefig(os.path.join(output_dir, "price_vs_year.png"))
plt.close()

# сохраняем график зависимости цены от пробега
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="mileage", y="price", alpha=0.5)
plt.title("Зависимость цены от пробега")
plt.savefig(os.path.join(output_dir, "price_vs_mileage.png"))
plt.close()

# сохраняем предобработанный датасет
df_encoded.to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)

print("Готово! Все файлы сохранены в:", output_dir)
