import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def prepare_training_data(df: pd.DataFrame):
    df = df.copy()

    # Ensure lat/lon columns exist and match app.py
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"}, errors="ignore")

    # Extract hour feature
    if "acq_datetime" in df.columns:
        df["hour"] = df["acq_datetime"].dt.hour
    else:
        df["hour"] = 0  # fallback if needed

    # Features must exactly match prediction
    features = ["brightness", "confidence", "frp", "lat", "lon", "hour"]
    target = "risk"

    # Drop rows with missing values for these columns
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    return X, y


def train_risk_model(df: pd.DataFrame):
    X, y = prepare_training_data(df)

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        random_state=42
    )
    model.fit(X, y)

    return model
