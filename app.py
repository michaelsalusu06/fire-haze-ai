import streamlit as st
import pandas as pd
import pydeck as pdk

from src.ml import train_risk_model
from src.data import (
    load_firms_24h,
    load_firms_7d,
    load_viirs_snpp_24h,
    load_viirs_noaa20_24h,
    filter_region,
    add_simple_risk,
    BOUNDS
)
from src.geo import color_from_risk
from src.air_quality import get_air_quality_data


# -------------------------------------------------
# STREAMLIT PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(page_title="🔥 Fire & Haze Watch", layout="wide")

# Header UI
st.title("🔥 Fire & Haze Watch — Sumatra, Kalimantan, Indonesia")
st.caption("Live NASA FIRMS (MODIS + VIIRS) — last 24h. Risk score boosted by AI model.")


# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
region = st.sidebar.selectbox("Region", ["Sumatra", "Kalimantan", "Indonesia"], index=0)
min_conf = st.sidebar.slider("Minimum Confidence", 0, 100, 50, step=5)
show_labels = st.sidebar.checkbox("Show Hotspot Details", value=True)
show_aqi = st.sidebar.checkbox("Show Air Quality Data")


# -------------------------------------------------
# TRAIN AI MODEL
# -------------------------------------------------
with st.spinner("Training AI Risk Model (MODIS 7-Day)..."):

    df_train = load_firms_7d()

    # MODIS 7-day has latitude/longitude already
    df_train = add_simple_risk(df_train)

    model = train_risk_model(df_train)

st.success("AI model trained successfully.")


# -------------------------------------------------
# LOAD FIRE DATA (MODIS + VIIRS)
# -------------------------------------------------
with st.spinner("Fetching live FIRMS data..."):

    df_modis = load_firms_24h()
    df_snpp = load_viirs_snpp_24h()
    df_noaa = load_viirs_noaa20_24h()

    dfs = [df_modis]
    if len(df_snpp) > 0:
        dfs.append(df_snpp)
    if len(df_noaa) > 0:
        dfs.append(df_noaa)

    # Common columns only
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    common_cols = list(common_cols)

    dfs = [d[common_cols] for d in dfs]

    df = pd.concat(dfs, ignore_index=True)

    df = filter_region(df, region)

    if len(df) == 0:
        st.warning("No hotspots found in this region.")
        st.stop()

    df = df[df["confidence"] >= min_conf]

    df = add_simple_risk(df)

    # Standardize naming
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})


# -------------------------------------------------
# AI MODEL PREDICTION
# -------------------------------------------------
df["hour"] = df["acq_datetime"].dt.hour

X_pred = df[["brightness", "confidence", "frp", "lat", "lon", "hour"]]
df["ai_risk"] = model.predict(X_pred)


# -------------------------------------------------
# SUMMARY METRICS
# -------------------------------------------------
col1, col2 = st.columns(2)
col1.metric("🔥 Hotspots", f"{len(df):,}")
col2.metric("📍 Region", region)


# -------------------------------------------------
# ANALYTICS SUMMARY
# -------------------------------------------------
with st.expander("📊 Analytics Summary", expanded=True):

    colA, colB, colC, colD = st.columns(4)

    colA.metric("Average Risk", f"{df['risk'].mean():.2f}")
    colB.metric("Average FRP", f"{df['frp'].mean():.1f}")
    colC.metric("High-Risk (%)", f"{(df['risk'] >= 4).mean() * 100:.1f}%")

    brightness_avg = df["brightness"].mean() if "brightness" in df.columns else 0
    colD.metric("Avg Brightness", f"{brightness_avg:.1f}")


# -------------------------------------------------
# RISK DISTRIBUTION CHART
# -------------------------------------------------
st.subheader("🔥 Hotspots by Risk Level")
risk_counts = df["risk"].value_counts().sort_index()
st.bar_chart(risk_counts)


# -------------------------------------------------
# TIME DISTRIBUTION
# -------------------------------------------------
st.subheader("⏱ Hotspots by Hour (Local Time)")
df["hour_local"] = df["acq_datetime"].dt.tz_convert("Asia/Jakarta").dt.hour
hour_counts = df["hour_local"].value_counts().sort_index()
st.line_chart(hour_counts)


# -------------------------------------------------
# AI TRAINING RESULTS
# -------------------------------------------------
with st.expander("🤖 AI Model Training Summary (7-day MODIS)"):

    st.write("### 🧠 Training Dataset (first 10 rows)")
    st.dataframe(df_train.head(10))

    st.write("### 🔥 Risk Distribution in Training Data")
    st.bar_chart(df_train["risk"].value_counts().sort_index())

    st.write("### 📈 Feature Importances")
    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": ["brightness", "confidence", "frp", "latitude", "longitude", "hour"],
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        st.dataframe(importances)
        st.bar_chart(importances.set_index("feature"))
    else:
        st.info("Model does not expose feature importances.")


# -------------------------------------------------
# COLOR HANDLING — SAFE VERSION
# -------------------------------------------------
df["color_hex"] = color_from_risk(df["risk"])

def safe_hex_to_rgb(h):
    if not isinstance(h, str):
        return [255, 255, 255]  # fallback white
    h = h.strip().lstrip("#")
    if len(h) != 6:
        return [255, 255, 255]  # fallback white
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

df["color"] = df["color_hex"].apply(safe_hex_to_rgb)


# -------------------------------------------------
# MAP VIEW
# -------------------------------------------------
bounds = BOUNDS[region]
mid_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2
mid_lon = (bounds["lon_min"] + bounds["lon_max"]) / 2

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[lon, lat]",
    get_fill_color="color",
    get_radius=4000,
    pickable=True,
    auto_highlight=True,
)

tooltip = {
    "html": (
        "<b>Confidence:</b> {confidence}<br/>"
        "<b>FRP:</b> {frp}<br/>"
        "<b>Brightness:</b> {brightness}<br/>"
        "<b>AI Risk:</b> {ai_risk}<br/>"
        "<b>Acquired:</b> {acq_datetime}<br/>"
        "<b>Risk (0-5):</b> {risk}"
    ),
    "style": {"backgroundColor": "rgba(20,20,20,0.9)", "color": "white"}
} if show_labels else None

view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=5.2)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip
))

st.markdown("**Legend:** 🟢 low, 🟡 medium, 🔴 high risk")


# -------------------------------------------------
# AIR QUALITY SECTION
# -------------------------------------------------
if show_aqi:
    st.subheader("🌫️ Current Air Quality in Indonesia (PM2.5)")
    aqi_df = get_air_quality_data()

    if len(aqi_df) == 0:
        st.info("No air quality data available at this time.")
    else:
        st.dataframe(aqi_df.head(15))

        if "pm25" in aqi_df.columns:
            st.metric("Average PM2.5 (µg/m³)", f"{aqi_df['pm25'].mean():.1f}")
        else:
            st.info("No PM2.5 readings available.")
