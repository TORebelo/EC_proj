import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
print("Libraries imported successfully.")
data = pd.read_csv('data/custom_covid19.csv') 
print("Data loaded with shape:", data.shape)
