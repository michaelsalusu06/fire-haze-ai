import pandas as pd
import requests

# ----------------------------
# MODIS (CSV)
# ----------------------------
FIRMS_7D = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/MODIS_C6_1_Global_7d.csv"
FIRMS_24H = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/MODIS_C6_1_Global_24h.csv"

# ----------------------------
# NEW VIIRS COUNTRY ENDPOINTS (ALWAYS WORK)
# ----------------------------
VIIRS_SNPP_JSON  = "https://firms.modaps.eosdis.nasa.gov/api/country/json/viirs-snpp/24h/IDN"
VIIRS_NOAA20_JSON = "https://firms.modaps.eosdis.nasa.gov/api/country/json/viirs-noaa20/24h/IDN"


def load_firms_7d() -> pd.DataFrame:
    df = pd.read_csv(FIRMS_7D)

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)

    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
        utc=True,
        errors="coerce"
    )
    return df

def load_viirs_snpp_24h() -> pd.DataFrame:
    try:
        r = requests.get(VIIRS_SNPP_JSON, timeout=10)
        data = r.json().get("features", [])
        df = pd.json_normalize([f["properties"] for f in data])
        return df
    except:
        return pd.DataFrame()


def load_viirs_noaa20_24h() -> pd.DataFrame:
    try:
        r = requests.get(VIIRS_NOAA20_JSON, timeout=10)
        data = r.json().get("features", [])
        df = pd.json_normalize([f["properties"] for f in data])
        return df
    except:
        return pd.DataFrame()


# ----------------------------
# REGION BOUNDS
# ----------------------------
BOUNDS = {
    "Sumatra":   {"lat_min": -6.5, "lat_max": 6.5,  "lon_min": 95.0,  "lon_max": 106.0},
    "Kalimantan":{"lat_min": -4.5, "lat_max": 3.0,  "lon_min":108.0,  "lon_max": 118.5},
    "Indonesia": {"lat_min":-11.0, "lat_max": 7.0,  "lon_min": 95.0,  "lon_max": 141.0},
}


# ----------------------------
# MODIS CSV
# ----------------------------
def load_firms_24h() -> pd.DataFrame:
    df = pd.read_csv(FIRMS_24H)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)

    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
        utc=True,
        errors="coerce"
    )
    return df


# ----------------------------
# FILTER REGION
# ----------------------------
def filter_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if "latitude" not in df.columns: return pd.DataFrame()
    if "longitude" not in df.columns: return pd.DataFrame()

    b = BOUNDS[region]
    return df[
        (df["latitude"].between(b["lat_min"], b["lat_max"])) &
        (df["longitude"].between(b["lon_min"], b["lon_max"]))
    ].reset_index(drop=True)


# ----------------------------
# RISK SCORE
# ----------------------------
def add_simple_risk(df: pd.DataFrame) -> pd.DataFrame:
    if "confidence" not in df.columns: df["confidence"] = 0
    if "frp" not in df.columns: df["frp"] = 0

    c = df["confidence"].fillna(0)
    frp = df["frp"].fillna(0)

    risk = (
        (c >= 30).astype(int) +
        (c >= 60).astype(int) +
        (c >= 85).astype(int) +
        (frp >= 30).astype(int) +
        (frp >= 80).astype(int)
    )

    df = df.copy()
    df["risk"] = risk.clip(0, 5)
    return df
