
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


