import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()
    df.drop(columns=["Person ID"], inplace=True, errors='ignore')
    df["Sleep Disorder"].fillna("No Disorder", inplace=True)

    # Label Encoding
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    le_bmi = LabelEncoder()
    df["BMI Category"] = le_bmi.fit_transform(df["BMI Category"])

    le_sleep = LabelEncoder()
    df["Sleep Disorder"] = le_sleep.fit_transform(df["Sleep Disorder"])

    # One-hot encoding for Occupation
    df = pd.get_dummies(df, columns=['Occupation'], dtype=int)

    # Blood Pressure split
    df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].astype(str).str.split('/', expand=True).astype(int)
    df.drop(columns=["Blood Pressure"], inplace=True)

    # Feature and label split
    X = df.drop(columns=["Sleep Disorder"])
    y = df["Sleep Disorder"]

    return X, y

def normalize_data(X):
    X_np = X.to_numpy().astype(float)
    X_min = np.min(X_np, axis=0)
    X_max = np.max(X_np, axis=0)
    X_scaled = (X_np - X_min) / (X_max - X_min + 1e-8)
    return X_scaled

def reshape_for_lstm(X_scaled):
    return X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
