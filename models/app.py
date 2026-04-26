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
import networkx as nx
import math
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


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
FIREBASE_BASE_URL = "https://loc-app-sathya-default-rtdb.asia-southeast1.firebasedatabase.app"

# ---------- Load ML Model ----------
@st.cache_resource
def load_ml_model():
    """Load the trained safety prediction model"""
    try:
        base_path = Path(__file__).resolve().parent
        model_path = base_path / 'safety_model.pkl'
        encoder_path = base_path / 'highway_encoder.pkl'
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        return model, encoder
    except FileNotFoundError:
        st.warning(
            "⚠️ ML model not found. Please ensure 'safety_model.pkl' and 'highway_encoder.pkl' exist in the models/ folder."
        )
        return None, None

# ---------- Fetch all users from Firebase ----------
@st.cache_data(ttl=10)
def fetch_all_users():
    """Fetch all users from Firebase"""
    try:
        resp = requests.get(f"{FIREBASE_BASE_URL}/users.json", timeout=5)
        if resp.status_code == 200:
            users = resp.json()
            if users:
                return list(users.keys())
    except Exception as e:
        st.error(f"Error fetching users: {e}")
    return []

# ---------- Fetch live location from Firebase ----------
@st.cache_data(ttl=10)
def fetch_live_location(user_id):
    """Fetch location for a specific user"""
    FIREBASE_URL = f"{FIREBASE_BASE_URL}/locations/{user_id}.json"
    try:
        resp = requests.get(FIREBASE_URL, timeout=5)
        loc = resp.json()
        if loc and "latitude" in loc and "longitude" in loc:
            return loc["latitude"], loc["longitude"], True
    except Exception as e:
        st.error(f"Error fetching location: {e}")
    return None, None, False

# ---------- Hazard Status Functions ----------
def update_hazard_status(user_id, status_value):
    """Update hazard status in Firebase (0=Safe, 1=Risk, 2=High Risk)"""
    try:
        FIREBASE_URL = f"{FIREBASE_BASE_URL}/hazards/{user_id}.json"
        # Send just the integer value to match mobile app format
        response = requests.put(FIREBASE_URL, json=status_value)
        
        if response.status_code == 200:
            return True, "Status updated successfully!"
        else:
            return False, f"Failed to update. Status code: {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_hazard_status(user_id):
    """Fetch current hazard status from Firebase"""
    try:
        FIREBASE_URL = f"{FIREBASE_BASE_URL}/hazards/{user_id}.json"
        response = requests.get(FIREBASE_URL)
        if response.status_code == 200:
            data = response.json()
            # Data is just an integer (0, 1, or 2), not an object
            if data is not None and isinstance(data, int):
                return data
        return None
    except:
        return None

# ---------- Road Attributes Function ----------
def get_road_attributes(lat, lon, dist=1000):
    try:
        G = ox.graph_from_point((lat, lon), dist=dist, network_type="drive")
        # OSMnx v2.x: ox.nearest_edges; v1.x: ox.distance.nearest_edges
        try:
            u, v, key = ox.nearest_edges(G, lon, lat)
        except AttributeError:
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
    
    now = datetime.now()
    hour_of_day = now.hour
    is_weekend = 1 if now.weekday() >= 5 else 0
    
    highway_type = road_info['highway']
    if isinstance(highway_type, list):
        highway_type = highway_type[0]
    
    if highway_type not in encoder.classes_:
        highway_type = 'unclassified'
    
    lanes = road_info['lanes']
    if not isinstance(lanes, int):
        try:
            lanes = int(lanes)
        except:
            lanes = 2
    
    highway_encoded = encoder.transform([highway_type])[0]

    feature_names = [
        'highway_encoded', 'lanes', 'maxspeed_kmph',
        'temperature', 'humidity', 'visibility', 'wind_speed',
        'hour_of_day', 'is_weekend'
    ]
    features = pd.DataFrame([[
        highway_encoded,
        lanes,
        road_info['maxspeed_kmph'],
        weather['temperature'],
        weather['humidity'],
        weather['visibility'],
        weather['wind_speed'],
        hour_of_day,
        is_weekend
    ]], columns=feature_names)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
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

# ---------- Route Computation Functions ----------
SAFETY_WEIGHTS = {'safe': 1, 'risk': 5, 'high_risk': 20}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

@st.cache_resource(show_spinner=False)
def get_route_graph(lat1, lon1, lat2, lon2):
    from shapely.geometry import box as shapely_box
    north = max(lat1, lat2) + 0.015
    south = min(lat1, lat2) - 0.015
    east  = max(lon1, lon2) + 0.015
    west  = min(lon1, lon2) - 0.015
    polygon = shapely_box(west, south, east, north)
    try:
        G = ox.graph_from_polygon(polygon, network_type='drive')
        return G, None
    except Exception as e:
        return None, str(e)

def compute_candidate_routes(G, orig_ll, dest_ll, k=3):
    try:
        # OSMnx v2.x uses ox.nearest_nodes; v1.x uses ox.distance.nearest_nodes
        try:
            orig_node = ox.nearest_nodes(G, orig_ll[1], orig_ll[0])
            dest_node = ox.nearest_nodes(G, dest_ll[1], dest_ll[0])
        except AttributeError:
            orig_node = ox.distance.nearest_nodes(G, orig_ll[1], orig_ll[0])
            dest_node = ox.distance.nearest_nodes(G, dest_ll[1], dest_ll[0])
        routes = []
        for path in nx.shortest_simple_paths(G, orig_node, dest_node, weight='length'):
            routes.append(path)
            if len(routes) >= k:
                break
        return routes
    except Exception:
        try:
            path = nx.shortest_path(G, orig_node, dest_node, weight='length')
            return [path]
        except:
            return []

def route_coords(G, path):
    coords = []
    for node in path:
        data = G.nodes[node]
        coords.append((data['y'], data['x']))
    return coords

def route_distance_km(coords):
    total = 0
    for i in range(len(coords)-1):
        total += haversine_km(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
    return total

def score_route(coords, model, encoder, weather_fetcher, sample_every_km=1.0):
    if model is None:
        return None, []
    scores = []
    segment_details = []
    dist_acc = 0.0
    sample_points = [coords[0]]
    for i in range(1, len(coords)):
        dist_acc += haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
        if dist_acc >= sample_every_km:
            sample_points.append(coords[i])
            dist_acc = 0.0
    if coords[-1] not in sample_points:
        sample_points.append(coords[-1])

    for lat, lon in sample_points:
        road_info = get_road_attributes(lat, lon, dist=300)
        weather   = weather_fetcher.fetch_weather(lat, lon)
        if road_info and weather:
            pred, probs = predict_safety(model, encoder, road_info, weather)
            if pred:
                w = SAFETY_WEIGHTS.get(pred, 5)
                scores.append(w)
                segment_details.append({'lat': lat, 'lon': lon, 'safety': pred, 'probs': probs})
    avg = sum(scores) / len(scores) if scores else 10
    return avg, segment_details

def classify_score(score):
    if score is None:   return 'unknown', '❓', 'gray'
    if score < 2:       return 'safe',     '✅', 'green'
    if score < 8:       return 'risk',     '⚠️', 'orange'
    return 'high_risk', '🚨', 'red'

def build_route_map(origin, destination, routes_data):
    mid_lat = (origin[0] + destination[0]) / 2
    mid_lon = (origin[1] + destination[1]) / 2
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=13)

    route_colors  = ['green', 'orange', 'red']
    route_weights = [6, 3, 3]
    route_dashes  = ['', '8 4', '4 4']

    for idx, rd in enumerate(routes_data):
        coords   = rd['coords']
        label    = rd['label']
        score    = rd['score']
        is_best  = idx == 0
        color    = route_colors[idx % len(route_colors)]
        weight   = route_weights[idx % len(route_weights)]
        dash     = route_dashes[idx % len(route_dashes)]
        _, icon_char, _ = classify_score(score)
        popup_txt = f"{label}<br>Score: {score:.1f} {icon_char}<br>Distance: {rd['dist_km']:.2f} km"
        folium.PolyLine(
            coords, color=color, weight=weight,
            dash_array=dash if dash else None,
            tooltip=label, popup=popup_txt, opacity=0.9 if is_best else 0.6
        ).add_to(m)

        for seg in rd.get('segments', []):
            _, _, seg_color = get_safety_color(seg['safety'])
            folium.CircleMarker(
                [seg['lat'], seg['lon']], radius=5,
                color=seg_color, fill=True, fill_opacity=0.7,
                popup=f"Safety: {seg['safety']}"
            ).add_to(m)

    folium.Marker(origin,      popup='🟢 Origin',      icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker(destination, popup='🔴 Destination',  icon=folium.Icon(color='red',   icon='stop')).add_to(m)
    return m

def build_google_maps_link(origin, destination, route_coords_list, max_waypoints=8):
    """Generate a shareable Google Maps Directions URL with intermediate waypoints."""
    orig_str = f"{origin[0]},{origin[1]}"
    dest_str = f"{destination[0]},{destination[1]}"

    # Sample evenly-spaced waypoints from the middle of the route (skip first/last)
    inner = route_coords_list[1:-1]
    if len(inner) > max_waypoints:
        step = len(inner) / max_waypoints
        inner = [inner[int(i * step)] for i in range(max_waypoints)]

    waypoints_str = "|".join(f"{lat},{lon}" for lat, lon in inner)

    base = "https://www.google.com/maps/dir/"
    url  = f"{base}?api=1&origin={orig_str}&destination={dest_str}&travelmode=driving"
    if waypoints_str:
        url += f"&waypoints={waypoints_str}"
    return url

def render_route_finder_tab(model, encoder, fetcher, live_lat=None, live_lon=None):
    st.markdown("## 🗺️ Safest Route Finder")
    st.markdown("Enter origin and destination coordinates to find the safest driving route.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🟢 Origin")
        use_live = False
        if live_lat and live_lon:
            use_live = st.checkbox("Use live tracked location as origin", value=True)
        orig_lat = st.number_input("Origin Latitude",  value=live_lat if (use_live and live_lat) else 13.0827, format="%.6f", key="orig_lat")
        orig_lon = st.number_input("Origin Longitude", value=live_lon if (use_live and live_lon) else 80.2707, format="%.6f", key="orig_lon")

    with col2:
        st.markdown("### 🔴 Destination")
        dest_lat = st.number_input("Destination Latitude",  value=13.0604, format="%.6f", key="dest_lat")
        dest_lon = st.number_input("Destination Longitude", value=80.2496, format="%.6f", key="dest_lon")

    dist_check = haversine_km(orig_lat, orig_lon, dest_lat, dest_lon)
    st.info(f"📏 Straight-line distance: **{dist_check:.2f} km**")
    if dist_check > 25:
        st.warning("⚠️ Distance > 25 km. Route computation may take longer (large OSM graph download).")

    if st.button("🔍 Find Safest Route", type="primary", use_container_width=True):
        if dist_check < 0.05:
            st.error("Origin and destination are too close or identical.")
            return

        with st.spinner("📡 Downloading road network…"):
            G, graph_err = get_route_graph(orig_lat, orig_lon, dest_lat, dest_lon)
        if G is None:
            st.error(f"❌ Could not download road network: {graph_err}")
            return

        with st.spinner("🔄 Computing candidate routes…"):
            paths = compute_candidate_routes(G, (orig_lat, orig_lon), (dest_lat, dest_lon), k=3)
        if not paths:
            st.error("❌ No route found between these coordinates.")
            return

        routes_data = []
        progress = st.progress(0, text="Scoring routes for safety…")
        for i, path in enumerate(paths):
            coords = route_coords(G, path)
            dist_km = route_distance_km(coords)
            score, segments = score_route(coords, model, encoder, fetcher)
            _, icon, _ = classify_score(score)
            label = f"Route {i+1} {icon}" if i > 0 else f"🏆 Safest Route {icon}"
            routes_data.append({'coords': coords, 'score': score, 'dist_km': dist_km,
                                 'segments': segments, 'label': label, 'path': path})
            progress.progress((i+1)/len(paths), text=f"Scored route {i+1}/{len(paths)}")

        routes_data.sort(key=lambda r: (r['score'] if r['score'] is not None else 999))
        routes_data[0]['label'] = f"🏆 Safest Route {classify_score(routes_data[0]['score'])[1]}"
        progress.empty()

        m = build_route_map((orig_lat, orig_lon), (dest_lat, dest_lon), routes_data)
        st_folium(m, width=None, height=600, returned_objects=[])

        st.markdown("### 📊 Route Comparison")
        table_rows = []
        for rd in routes_data:
            rating, icon, _ = classify_score(rd['score'])
            speed_kmh = 40
            eta_min = (rd['dist_km'] / speed_kmh) * 60
            table_rows.append({
                "Route":         rd['label'],
                "Distance (km)": f"{rd['dist_km']:.2f}",
                "Est. Time":     f"{eta_min:.0f} min",
                "Safety Score":  f"{rd['score']:.1f}" if rd['score'] else "N/A",
                "Rating":        f"{icon} {rating.replace('_',' ').title()}"
            })
        st.table(pd.DataFrame(table_rows))

        best = routes_data[0]
        rating, icon, _ = classify_score(best['score'])
        st.success(f"{icon} **Recommended: {best['label']}** — {best['dist_km']:.2f} km, safety rating: {rating.replace('_',' ').title()}")

        # ---------- Google Maps Share Link ----------
        st.markdown("### 📲 Share Safe Route")
        gmap_url = build_google_maps_link(
            (orig_lat, orig_lon),
            (dest_lat, dest_lon),
            best['coords'],
            max_waypoints=8
        )
        share_col1, share_col2 = st.columns([3, 1])
        with share_col1:
            st.code(gmap_url, language=None)
        with share_col2:
            st.link_button("🗺️ Open in Google Maps", gmap_url, use_container_width=True, type="primary")
        st.caption(
            "⚠️ Google Maps may adjust the route slightly based on real-time traffic. "
            "The waypoints approximate the safest computed path."
        )
        st.markdown("---")

        if best['segments']:
            with st.expander("🔍 Segment-level safety details (safest route)"):
                seg_df = pd.DataFrame([{
                    'Lat': f"{s['lat']:.5f}", 'Lon': f"{s['lon']:.5f}",
                    'Safety': s['safety'],
                    'Safe%': f"{s['probs'].get('safe',0)*100:.1f}%",
                    'Risk%': f"{s['probs'].get('risk',0)*100:.1f}%",
                    'HighRisk%': f"{s['probs'].get('high_risk',0)*100:.1f}%"
                } for s in best['segments']])
                st.dataframe(seg_df, use_container_width=True)

# ---------- Main ----------
st.title("🗺️ Live Location Tracker with AI Safety Prediction")

# Initialize session state
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None

# Load ML model
model, encoder = load_ml_model()
fetcher = WeatherFetcher(API_KEY)

# Fetch all users
all_users = fetch_all_users()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("👥 Users")
    
    # Display all users as clickable buttons
    if all_users:
        for user in all_users:
            # Get hazard status for visual indicator
            hazard_status = get_hazard_status(user)
            status_indicator = ""
            if hazard_status == 0:
                status_indicator = "✅"
            elif hazard_status == 1:
                status_indicator = "⚠️"
            elif hazard_status == 2:
                status_indicator = "🚨"
            
            # Create button for each user
            is_selected = st.session_state.selected_user == user
            if st.button(
                f"{status_indicator} {user}", 
                key=f"user_{user}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_user = user
                st.rerun()
    else:
        st.info("No users found in database")
    
    st.markdown("---")
    
    # Show details for selected user
    if st.session_state.selected_user:
        selected_user = st.session_state.selected_user
        live_lat, live_lon, success = fetch_live_location(selected_user)
        
        if success:
            st.header(f"📍 Live Data - {selected_user}")
            st.write(f"**Latitude:** {live_lat:.6f}")
            st.write(f"**Longitude:** {live_lon:.6f}")
            st.write(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")

            road_info = get_road_attributes(live_lat, live_lon)
            weather = fetcher.fetch_weather(live_lat, live_lon)

            # Predict safety
            safety_prediction = None
            safety_probs = None
            if model and road_info and weather:
                safety_prediction, safety_probs = predict_safety(model, encoder, road_info, weather)

            # Safety Prediction Display
            if safety_prediction:
                st.markdown("---")
                color_hex, icon, _ = get_safety_color(safety_prediction)
                st.markdown(f"### {icon} Safety Status")
                st.markdown(f"<h2 style='color: {color_hex};'>{safety_prediction.upper().replace('_', ' ')}</h2>", 
                           unsafe_allow_html=True)
                
                st.write("**Confidence Levels:**")
                for level in ['safe', 'risk', 'high_risk']:
                    prob = safety_probs.get(level, 0)
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

# ---------------- Main Content (Tabs) ----------------
tab1, tab2 = st.tabs(["📍 Live Tracker", "🗺️ Safest Route Finder"])

with tab2:
    live_lat_for_route, live_lon_for_route = None, None
    if st.session_state.selected_user:
        _lat, _lon, _ok = fetch_live_location(st.session_state.selected_user)
        if _ok:
            live_lat_for_route, live_lon_for_route = _lat, _lon
    render_route_finder_tab(model, encoder, fetcher, live_lat_for_route, live_lon_for_route)

with tab1:
    if st.session_state.selected_user:
        selected_user = st.session_state.selected_user
        live_lat, live_lon, success = fetch_live_location(selected_user)

        if success:
            road_info = get_road_attributes(live_lat, live_lon)
            weather = fetcher.fetch_weather(live_lat, live_lon)

            safety_prediction = None
            safety_probs = None
            if model and road_info and weather:
                safety_prediction, safety_probs = predict_safety(model, encoder, road_info, weather)

            # Map and Buttons Layout
            map_col, button_col = st.columns([4, 1])

            with map_col:
                m = folium.Map(location=[live_lat, live_lon], zoom_start=17)

                if safety_prediction:
                    _, icon, marker_color = get_safety_color(safety_prediction)
                    folium.Marker(
                        [live_lat, live_lon],
                        popup=f"User: {selected_user}<br>Safety: {safety_prediction}",
                        icon=folium.Icon(color=marker_color, icon='info-sign')
                    ).add_to(m)
                else:
                    folium.Marker([live_lat, live_lon], popup=f"User: {selected_user}").add_to(m)

                st_folium(m, width=None, height=700, returned_objects=[])

            with button_col:
                st.markdown("### 🚨 Report Hazard")

                current_hazard = get_hazard_status(selected_user)
                status_map = {0: "✅ SAFE", 1: "⚠️ RISK", 2: "🚨 HIGH RISK"}
                if current_hazard is not None:
                    st.info(f"**Current:** {status_map.get(current_hazard, 'Unknown')}")

                if 'update_message' not in st.session_state:
                    st.session_state.update_message = None

                if st.button("✅ SAFE", key="hazard_safe", use_container_width=True, type="secondary"):
                    success_update, message = update_hazard_status(selected_user, 0)
                    st.session_state.update_message = ("success" if success_update else "error", message)
                    if success_update: st.cache_data.clear()

                if st.button("⚠️ RISK", key="hazard_risk", use_container_width=True, type="secondary"):
                    success_update, message = update_hazard_status(selected_user, 1)
                    st.session_state.update_message = ("warning" if success_update else "error", message)
                    if success_update: st.cache_data.clear()

                if st.button("🚨 HIGH RISK", key="hazard_high", use_container_width=True, type="secondary"):
                    success_update, message = update_hazard_status(selected_user, 2)
                    st.session_state.update_message = ("error", message)
                    if success_update: st.cache_data.clear()

                if st.session_state.update_message:
                    msg_type, msg_text = st.session_state.update_message
                    if msg_type == "success": st.success(msg_text)
                    elif msg_type == "warning": st.warning(msg_text)
                    else: st.error(msg_text)
                    st.session_state.update_message = None

                st.caption("Report current location's safety status")

                if safety_prediction:
                    st.markdown("---")
                    st.markdown("### 🤖 AI Prediction")
                    safe_prob     = safety_probs.get('safe', 0) * 100
                    risk_prob     = safety_probs.get('risk', 0) * 100
                    high_risk_prob = safety_probs.get('high_risk', 0) * 100
                    st.button(f"✅ {safe_prob:.1f}%",      key="ai_safe", use_container_width=True, disabled=True)
                    st.button(f"⚠️ {risk_prob:.1f}%",      key="ai_risk", use_container_width=True, disabled=True)
                    st.button(f"🚨 {high_risk_prob:.1f}%", key="ai_high", use_container_width=True, disabled=True)

            # Safety Details Card
            if safety_prediction:
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AI Safety Prediction", safety_prediction.replace('_', ' ').title())
                with col2:
                    confidence = max(safety_probs.values()) * 100
                    st.metric("AI Confidence", f"{confidence:.1f}%")
                with col3:
                    risk_factors = 0
                    if weather['visibility'] < 1000:  risk_factors += 1
                    if weather['wind_speed'] > 15:    risk_factors += 1
                    if road_info['maxspeed_kmph'] > 80: risk_factors += 1
                    if weather['temperature'] < 5 or weather['temperature'] > 35: risk_factors += 1
                    st.metric("Risk Factors", risk_factors)

            if st.button("🔄 Refresh Location"):
                st.cache_data.clear()
                st.rerun()

        else:
            st.error(f"❌ Could not fetch location for user {selected_user}")
    else:
        st.info("👈 Please select a user from the sidebar to view their location")


# Footer
st.markdown("---")
st.caption("🤖 Powered by Machine Learning | Real-time location tracking with AI-based safety prediction")