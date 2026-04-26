import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
import time
import joblib
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Live Location Tracker with Safety AI")

# ----------------- CONFIG -----------------
DEFAULT_SPEED_MAP = {
    "motorway": 100,
    "trunk": 80,
    "primary": 70,
    "secondary": 60,
    "tertiary": 50,
    "residential": 40,
    "service": 30,
    "unclassified": 35,
}

API_KEY = "ca490bf9719a27199d0cba47047951a9"

# ---------- Load ML Model ----------
@st.cache_resource
def load_ml_model():
    """Load the trained safety prediction model"""
    try:
        model = joblib.load('safety_model.pkl')
        encoder = joblib.load('highway_encoder.pkl')
        return model, encoder
    except FileNotFoundError:
        st.warning("⚠️ ML model not found. Please train the model first using the training script.")
        return None, None

# ---------- Fetch live location from Firebase ----------
@st.cache_data(ttl=10)
def fetch_live_location():
    FIREBASE_URL = "https://loc-app-sathya-default-rtdb.asia-southeast1.firebasedatabase.app/locations/9080858902.json"
    try:
        resp = requests.get(FIREBASE_URL, timeout=5)
        loc = resp.json()
        if loc and "latitude" in loc and "longitude" in loc:
            return loc["latitude"], loc["longitude"], True
    except Exception as e:
        st.error(f"Error fetching location: {e}")
    return None, None, False

# ---------- Road Attributes Function ----------
def get_road_attributes(lat, lon, dist=1000):
    try:
        G = ox.graph_from_point((lat, lon), dist=dist, network_type="drive")
        u, v, key = ox.distance.nearest_edges(G, lon, lat)
        edge_data = G[u][v][key]

        highway_type = edge_data.get("highway", "unknown")
        lanes = edge_data.get("lanes", "unknown")
        maxspeed = edge_data.get("maxspeed", None)

        if isinstance(maxspeed, list): maxspeed = maxspeed[0]
        if isinstance(maxspeed, str): maxspeed = maxspeed.replace(" km/h", "").replace("mph", "").strip()

        if not maxspeed or pd.isna(maxspeed):
            maxspeed = DEFAULT_SPEED_MAP.get(highway_type, 50)
        else:
            try: maxspeed = int(maxspeed)
            except: maxspeed = DEFAULT_SPEED_MAP.get(highway_type, 50)

        return {
            "latitude": lat,
            "longitude": lon,
            "highway": highway_type,
            "lanes": lanes if isinstance(lanes, int) else 2,
            "maxspeed_kmph": maxspeed,
        }
    except Exception as e:
        print("⚠️ Error fetching OSM data:", e)
        return None

# ---------- Weather Fetcher ----------
class WeatherFetcher:
    def __init__(self, api_key, min_interval_sec=87):
        self.api_key = api_key
        self.min_interval = min_interval_sec
        self.last_fetch_time = 0
        self.cached_weather = None

    def fetch_weather(self, lat, lon):
        current_time = time.time()
        if self.cached_weather and (current_time - self.last_fetch_time) < self.min_interval:
            return self.cached_weather

        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code == 200:
            weather_info = {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "visibility": data.get("visibility", 5000),
                "weather": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
            self.cached_weather = weather_info
            self.last_fetch_time = current_time
            return weather_info
        else:
            return self.cached_weather

# ---------- Safety Prediction Function ----------
def predict_safety(model, encoder, road_info, weather):
    """Predict safety level using ML model"""
    if model is None or encoder is None:
        return None, None
    
    # Get current time info
    now = datetime.now()
    hour_of_day = now.hour
    is_weekend = 1 if now.weekday() >= 5 else 0
    
    # Extract features
    highway_type = road_info['highway']
    if isinstance(highway_type, list):
        highway_type = highway_type[0]
    
    # Handle unknown highway types
    if highway_type not in encoder.classes_:
        highway_type = 'unclassified'
    
    lanes = road_info['lanes']
    if not isinstance(lanes, int):
        try:
            lanes = int(lanes)
        except:
            lanes = 2
    
    # Encode highway type
    highway_encoded = encoder.transform([highway_type])[0]
    
    # Prepare features
    features = np.array([[
        highway_encoded,
        lanes,
        road_info['maxspeed_kmph'],
        weather['temperature'],
        weather['humidity'],
        weather['visibility'],
        weather['wind_speed'],
        hour_of_day,
        is_weekend
    ]])
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Create probability dictionary
    prob_dict = {
        class_name: float(prob) 
        for class_name, prob in zip(model.classes_, probabilities)
    }
    
    return prediction, prob_dict

# ---------- Safety Color Mapping ----------
def get_safety_color(safety_level):
    """Return color and icon for safety level"""
    colors = {
        'safe': ('#28a745', '✅', 'green'),
        'risk': ('#ffc107', '⚠️', 'orange'),
        'high_risk': ('#dc3545', '🚨', 'red')
    }
    return colors.get(safety_level, ('#6c757d', '❓', 'gray'))

# ---------- Main ----------
st.title("🗺️ Live Location Tracker with AI Safety Prediction")

# Load ML model
model, encoder = load_ml_model()

fetcher = WeatherFetcher(API_KEY)
live_lat, live_lon, success = fetch_live_location()

if success:
    road_info = get_road_attributes(live_lat, live_lon)
    weather = fetcher.fetch_weather(live_lat, live_lon)

    # Predict safety
    safety_prediction = None
    safety_probs = None
    if model and road_info and weather:
        safety_prediction, safety_probs = predict_safety(model, encoder, road_info, weather)

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("📍 Live Data")
        st.write(f"**Latitude:** {live_lat:.6f}")
        st.write(f"**Longitude:** {live_lon:.6f}")
        st.write(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")

        # Safety Prediction Display
        if safety_prediction:
            st.markdown("---")
            color_hex, icon, _ = get_safety_color(safety_prediction)
            st.markdown(f"### {icon} Safety Status")
            st.markdown(f"<h2 style='color: {color_hex};'>{safety_prediction.upper().replace('_', ' ')}</h2>", 
                       unsafe_allow_html=True)
            
            # Probability bars
            st.write("**Confidence Levels:**")
            for level in ['safe', 'risk', 'high_risk']:
                prob = safety_probs.get(level, 0)
                color = get_safety_color(level)[0]
                st.markdown(f"**{level.replace('_', ' ').title()}:** {prob:.1%}")
                st.progress(prob)
        
        st.markdown("---")

        if road_info:
            st.subheader("🛣️ Road Info")
            st.write(f"**Highway:** {road_info['highway']}")
            st.write(f"**Lanes:** {road_info['lanes']}")
            st.write(f"**Max Speed:** {road_info['maxspeed_kmph']} km/h")

        if weather:
            st.subheader("🌦️ Weather Info")
            st.write(f"**Temperature:** {weather['temperature']}°C")
            st.write(f"**Humidity:** {weather['humidity']}%")
            st.write(f"**Visibility:** {weather['visibility']} m")
            st.write(f"**Condition:** {weather['weather']}")
            st.write(f"**Wind Speed:** {weather['wind_speed']} m/s")

    # ---------------- Map and Buttons Layout ----------------
    map_col, button_col = st.columns([4, 1])
    
    with map_col:
        # Store map in session state to prevent recreation
        if 'map_rendered' not in st.session_state:
            st.session_state.map_rendered = True
        
        m = folium.Map(location=[live_lat, live_lon], zoom_start=17)
        
        # Add marker with safety color
        if safety_prediction:
            _, icon, marker_color = get_safety_color(safety_prediction)
            folium.Marker(
                [live_lat, live_lon],
                popup=f"Safety: {safety_prediction}",
                icon=folium.Icon(color=marker_color, icon='info-sign')
            ).add_to(m)
        else:
            folium.Marker([live_lat, live_lon]).add_to(m)
        
        # Return value prevents rerun on map interaction
        st_folium(m, width=None, height=700, returned_objects=[])
    
    with button_col:
        st.markdown("### 🚦 Safety Levels")
        
        if safety_prediction:
            safe_prob = safety_probs.get('safe', 0) * 100
            risk_prob = safety_probs.get('risk', 0) * 100
            high_risk_prob = safety_probs.get('high_risk', 0) * 100
            
            # Safe button (returns 0)
            is_safe = safety_prediction == 'safe'
            if st.button(f"✅ SAFE\n{safe_prob:.1f}%", 
                        key="safe_btn_right", 
                        use_container_width=True,
                        type="primary" if is_safe else "secondary"):
                print("0")  # Print to terminal
                print(f"Safety Level Selected: SAFE (0)")
            
            st.write("")  # Spacing
            
            # Risk button (returns 1)
            is_risk = safety_prediction == 'risk'
            if st.button(f"⚠️ RISK\n{risk_prob:.1f}%", 
                        key="risk_btn_right", 
                        use_container_width=True,
                        type="primary" if is_risk else "secondary"):
                print("1")  # Print to terminal
                print(f"Safety Level Selected: RISK (1)")
            
            st.write("")  # Spacing
            
            # High Risk button (returns 2)
            is_high_risk = safety_prediction == 'high_risk'
            if st.button(f"🚨 HIGH RISK\n{high_risk_prob:.1f}%", 
                        key="high_risk_btn_right", 
                        use_container_width=True,
                        type="primary" if is_high_risk else "secondary"):
                print("2")  # Print to terminal
                print(f"Safety Level Selected: HIGH_RISK (2)")

    # Safety Details Card with Buttons
    if safety_prediction:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Safety", safety_prediction.replace('_', ' ').title())
        
        with col2:
            confidence = max(safety_probs.values()) * 100
            st.metric("AI Confidence", f"{confidence:.1f}%")
        
        with col3:
            risk_factors = 0
            if weather['visibility'] < 1000: risk_factors += 1
            if weather['wind_speed'] > 15: risk_factors += 1
            if road_info['maxspeed_kmph'] > 80: risk_factors += 1
            if weather['temperature'] < 5 or weather['temperature'] > 35: risk_factors += 1
            st.metric("Risk Factors", risk_factors)
        
        # Safety Level Buttons
        st.markdown("### 🚦 Safety Level Overview")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            safe_prob = safety_probs.get('safe', 0) * 100
            is_safe = safety_prediction == 'safe'
            btn_type = "primary" if is_safe else "secondary"
            st.button(
                f"✅ SAFE\n{safe_prob:.1f}%", 
                type=btn_type,
                use_container_width=True,
                key="safe_btn"
            )
        
        with btn_col2:
            risk_prob = safety_probs.get('risk', 0) * 100
            is_risk = safety_prediction == 'risk'
            btn_type = "primary" if is_risk else "secondary"
            st.button(
                f"⚠️ RISK\n{risk_prob:.1f}%", 
                type=btn_type,
                use_container_width=True,
                key="risk_btn"
            )
        
        with btn_col3:
            high_risk_prob = safety_probs.get('high_risk', 0) * 100
            is_high_risk = safety_prediction == 'high_risk'
            btn_type = "primary" if is_high_risk else "secondary"
            st.button(
                f"🚨 HIGH RISK\n{high_risk_prob:.1f}%", 
                type=btn_type,
                use_container_width=True,
                key="high_risk_btn"
            )

    if st.button("🔄 Refresh Location"):
        st.cache_data.clear()
        st.rerun()
else:
    st.error("❌ Could not fetch live location")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Machine Learning | Real-time location tracking with AI-based safety prediction")