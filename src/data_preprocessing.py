import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled