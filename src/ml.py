import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def prepare_training_data(df: pd.DataFrame):
    df = df.copy()

    # --- FIX LAT/LON COLUMN ISSUES ---
    # Ensure latitude → lat
    if "lat" not in df.columns:
        if "latitude" in df.columns:
            df["lat"] = df["latitude"]
        else:
            df["lat"] = 0  # fallback if missing

    # Ensure longitude → lon
    if "lon" not in df.columns:
        if "longitude" in df.columns:
            df["lon"] = df["longitude"]
        else:
            df["lon"] = 0  # fallback if missing

    # Extract hour feature
    if "acq_datetime" in df.columns:
        df["hour"] = df["acq_datetime"].dt.hour
    else:
        df["hour"] = 0

    # Features MUST match exactly with prediction step in app.py
    features = ["brightness", "confidence", "frp", "lat", "lon", "hour"]
    target = "risk"

    # Remove rows missing mandatory ML columns
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
