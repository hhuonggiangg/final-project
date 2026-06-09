# 🌐 PART 3: STREAMLIT WEBAPP - REDESIGNED
# Spotify Song Popularity Prediction - Gen-Z Aesthetic Edition
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG & THEME
# ============================================================================

st.set_page_config(
    page_title="🎵 Spotify Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Gen-Z Aesthetic Design System
st.markdown("""
<style>
:root {
    --primary: #7C3AED;
    --secondary: #EC4899;
    --accent: #06B6D4;
    --dark: #0F172A;
    --light: #F8FAFC;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    color: white;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.main {
    background: transparent;
    padding: 0;
}

.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 2rem;
    transition: all 0.3s ease;
}

.glass-card:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(124, 58, 237, 0.3);
    transform: translateY(-2px);
}

/* Gradient Text */
.gradient-text {
    background: linear-gradient(135deg, #7C3AED 0%, #EC4899 50%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Modern Buttons */
.btn-primary {
    background: linear-gradient(135deg, #7C3AED, #EC4899);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 8px 20px rgba(236, 72, 153, 0.3);
}

.btn-primary:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(236, 72, 153, 0.4);
}

.btn-secondary {
    background: rgba(255, 255, 255, 0.1);
    border: 1.5px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-secondary:hover {
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
}

/* Metric Cards */
.metric-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(124, 58, 237, 0.3);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7C3AED, #EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.875rem;
    margin-top: 0.5rem;
}

/* Slider Styling */
[data-testid="stSlider"] > label {
    color: white !important;
}

[data-testid="stSlider"] > div > div {
    background: linear-gradient(90deg, rgba(124,58,237,0.3), rgba(236,72,153,0.3), rgba(6,182,212,0.3));
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
}

/* Text & Headers */
h1, h2, h3 {
    color: white;
    font-weight: 700;
}

/* Tags */
.tag {
    display: inline-block;
    background: rgba(124, 58, 237, 0.2);
    border: 1px solid rgba(124, 58, 237, 0.4);
    color: #06B6D4;
    padding: 0.4rem 0.8rem;
    border-radius: 999px;
    font-size: 0.875rem;
    margin: 0.25rem;
    font-weight: 500;
}

/* Success/Info Badges */
.success-badge {
    background: rgba(34, 197, 94, 0.2);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #22C55E;
}

.info-badge {
    background: rgba(6, 182, 212, 0.2);
    border: 1px solid rgba(6, 182, 212, 0.4);
    color: #06B6D4;
}

.warning-badge {
    background: rgba(249, 115, 22, 0.2);
    border: 1px solid rgba(249, 115, 22, 0.4);
    color: #F97316;
}

/* Animations */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.animate-in {
    animation: slideIn 0.6s ease-out;
}

.animate-pulse {
    animation: pulse 2s infinite;
}

/* Container */
.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

/* Responsive */
@media (max-width: 768px) {
    .glass-card {
        padding: 1.5rem;
    }
    
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.5rem !important; }
}
</style>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE & INITIALIZATION
# ============================================================================

if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'username': 'Guest User',
        'predictions_made': 0,
        'accuracy_score': 0,
        'favorites': [],
        'badges': []
    }

if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None

if 'leaderboard' not in st.session_state:
    st.session_state.leaderboard = pd.DataFrame({
        'Username': ['DJ Vibes', 'Music Guru', 'Beat Master', 'Sound Prophet', 'Melody Maker'],
        'Predictions': [145, 128, 112, 98, 87],
        'Accuracy': [87, 84, 81, 79, 76],
        'Shares': [34, 28, 25, 19, 14]
    })

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('spotify_songs.csv')
        return df
    except:
        st.error("❌ Could not load data. Make sure spotify_songs.csv exists!")
        return None

@st.cache_resource
def load_models():
    try:
        with open('model_lr.pkl', 'rb') as f:
            model_lr = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model_lr, scaler, model_info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

try:
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except:
    feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 
                     'tempo', 'explicit', 'duration_ms']

df = load_data()
model_lr, model_rf, scaler, model_info = load_models()

if model_lr is None:
    st.warning("⚠️ Models not found! Run Part 1 & 2 first to train models.")
    st.info("Steps:\n1. Run: jupyter notebook notebooks/01_EDA.ipynb\n2. Run: jupyter notebook notebooks/02_Model_Training.ipynb\n3. Then run this app again!")
    st.stop()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_share_image(score, title, artist, mood_tags):
    """Generate shareable image for Instagram"""
    img = Image.new('RGB', (1080, 1920), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    
    # Gradient background (simulated with shapes)
    for i in range(1920):
        r = int(124 + (236-124) * (i/1920))
        g = int(58 + (72-58) * (i/1920))
        b = int(237 + (153-237) * (i/1920))
        draw.rectangle([(0, i), (1080, i+1)], fill=(r, g, b))
    
    # Score circle
    draw.ellipse([340, 400, 740, 800], outline=(255, 255, 255), width=8)
    draw.text((540, 580), f"{int(score)}", fill=(255, 255, 255), anchor="mm")
    draw.text((540, 650), "HIT SCORE", fill=(255, 255, 255), anchor="mm")
    
    # Title & Artist
    draw.text((540, 900), title, fill=(255, 255, 255), anchor="mm")
    draw.text((540, 980), f"by {artist}", fill=(200, 200, 200), anchor="mm")
    
    # Tags
    tag_y = 1100
    for tag in mood_tags[:3]:
        draw.text((540, tag_y), f"#{tag}", fill=(6, 182, 212), anchor="mm")
        tag_y += 80
    
    draw.text((540, 1800), "🎵 Predicted with Spotify Predictor", fill=(150, 150, 150), anchor="mm")
    
    return img

def get_mood_emoji(score):
    """Get emoji based on score"""
    if score >= 80:
        return "🔥"
    elif score >= 70:
        return "👍"
    elif score >= 60:
        return "😐"
    else:
        return "📉"

def get_accuracy_badge(accuracy):
    """Get badge based on accuracy"""
    if accuracy >= 85:
        return "⭐⭐⭐⭐⭐"
    elif accuracy >= 75:
        return "⭐⭐⭐⭐"
    elif accuracy >= 65:
        return "⭐⭐⭐"
    elif accuracy >= 50:
        return "⭐⭐"
    else:
        return "⭐"

def create_interactive_chart(data, x, y, title):
    """Create interactive Plotly chart"""
    fig = px.scatter(data, x=x, y=y, title=title,
                    labels={x: x.capitalize(), y: y.capitalize()},
                    color=y, size=y,
                    color_continuous_scale="Viridis")
    
    fig.update_layout(
        template="plotly_dark",
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0.1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        height=500
    )
    return fig

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h1 style='font-size: 2.5rem;'>🎵</h1>
    <h2 style='background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        Spotify Predictor
    </h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 Predict", "📊 Analytics", "🎯 Database", "🏆 Leaderboard", "👤 Profile"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# User Profile Mini Card in Sidebar
st.sidebar.markdown(f"""
<div style='background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius: 16px; padding: 1rem; border: 1px solid rgba(124,58,237,0.3); margin: 1rem 0;'>
    <div style='text-align: center;'>
        <h4 style='margin-bottom: 0.5rem;'>👤 {st.session_state.user_data['username']}</h4>
        <p style='color: rgba(255,255,255,0.6); margin: 0.25rem 0;'>Predictions: {st.session_state.user_data['predictions_made']}</p>
        <p style='color: rgba(255,255,255,0.6); margin: 0.25rem 0;'>Accuracy: {st.session_state.user_data['accuracy_score']}%</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
**About This App:**

🎨 Modern Gen-Z Design  
🚀 Interactive Features  
📱 Mobile Optimized  
🌐 Share Ready  
""")

# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown("""
    <div style='animation: slideIn 0.6s ease-out;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>
            <span class='gradient-text'>🔮 Discover What Makes Songs HIT</span>
        </h1>
        <p style='font-size: 1.1rem; color: rgba(255,255,255,0.7); margin-bottom: 2rem;'>
            AI-powered music analytics | Predict popularity | Share your predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{model_info['test_r2_lr']*100:.1f}%</div>
            <div class='metric-label'>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>±{model_info['test_mae_lr']:.1f}</div>
            <div class='metric-label'>Avg Error</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{len(df)}</div>
            <div class='metric-label'>Songs DB</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{df['artist_name'].nunique()}</div>
            <div class='metric-label'>Artists</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Section
    st.markdown("<h2 style='margin-top: 2rem; margin-bottom: 1.5rem;'>✨ Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='color: #06B6D4; margin-bottom: 1rem;'>🔮 Smart Prediction</h3>
            <p>Predict song popularity using advanced ML models trained on real Spotify data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='color: #EC4899; margin-bottom: 1rem;'>📱 Share Everything</h3>
            <p>Generate beautiful cards and share directly to Instagram, Twitter & Email</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='color: #7C3AED; margin-bottom: 1rem;'>🏆 Compete & Earn</h3>
            <p>Join leaderboards, collect badges, and compete with music enthusiasts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Trending Section
    st.markdown("<h2 style='margin-top: 2rem; margin-bottom: 1.5rem;'>🔥 Trending Predictions</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    sample_songs = df.sample(5)
    for col, (_, song) in zip([col1, col2, col3, col4, col5], sample_songs.iterrows()):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center;'>
                <h4 style='margin-bottom: 0.5rem; color: #06B6D4;'>{song['track_name'][:20]}...</h4>
                <p style='color: rgba(255,255,255,0.6); font-size: 0.85rem; margin: 0.5rem 0;'>{song['artist_name'][:15]}...</p>
                <div style='font-size: 2rem; margin: 1rem 0;'>{get_mood_emoji(song['popularity'])}</div>
                <p style='background: linear-gradient(135deg, #7C3AED, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.5rem; font-weight: 700;'>{int(song['popularity'])}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("<h2 style='margin-top: 2rem; margin-bottom: 1.5rem;'>📈 Database Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Popularity", f"{df['popularity'].mean():.1f}/100")
    with col2:
        st.metric("Avg Energy", f"{df['energy'].mean():.2f}/1.0")
    with col3:
        st.metric("Avg Danceability", f"{df['danceability'].mean():.2f}/1.0")
    with col4:
        st.metric("Avg Valence", f"{df['valence'].mean():.2f}/1.0")

# ============================================================================
# PAGE: PREDICT POPULARITY
# ============================================================================

elif page == "🔮 Predict":
    st.markdown("""
    <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
        <span class='gradient-text'>🔮 Will Your Song HIT?</span>
    </h1>
    <p style='color: rgba(255,255,255,0.7); margin-bottom: 2rem;'>Adjust features and get instant AI predictions</p>
    """, unsafe_allow_html=True)
    
    # Preset Buttons
    st.markdown("<h3 style='margin: 1.5rem 0;'>⚡ Quick Presets</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    presets = {
        '☀️ Summer Hit': {'dance': 0.8, 'energy': 0.85, 'loud': -3, 'speech': 0.03, 'acoust': 0.05, 'instrum': 0, 'live': 0.1, 'valence': 0.8, 'tempo': 120, 'explicit': 0},
        '😎 Chill Vibes': {'dance': 0.5, 'energy': 0.3, 'loud': -8, 'speech': 0.02, 'acoust': 0.5, 'instrum': 0.1, 'live': 0.08, 'valence': 0.6, 'tempo': 90, 'explicit': 0},
        '🎉 Party Anthem': {'dance': 0.9, 'energy': 0.95, 'loud': 0, 'speech': 0.04, 'acoust': 0.02, 'instrum': 0, 'live': 0.15, 'valence': 0.75, 'tempo': 130, 'explicit': 1},
        '💔 Sad Boi': {'dance': 0.4, 'energy': 0.3, 'loud': -10, 'speech': 0.02, 'acoust': 0.6, 'instrum': 0.05, 'live': 0.1, 'valence': 0.2, 'tempo': 70, 'explicit': 0},
    }
    
    preset_selected = None
    
    with col1:
        if st.button("☀️ Summer Hit", use_container_width=True, key="preset_summer"):
            preset_selected = '☀️ Summer Hit'
    with col2:
        if st.button("😎 Chill Vibes", use_container_width=True, key="preset_chill"):
            preset_selected = '😎 Chill Vibes'
    with col3:
        if st.button("🎉 Party Anthem", use_container_width=True, key="preset_party"):
            preset_selected = '🎉 Party Anthem'
    with col4:
        if st.button("💔 Sad Boi", use_container_width=True, key="preset_sad"):
            preset_selected = '💔 Sad Boi'
    
    # Main Prediction Section
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("<h3 style='margin: 1.5rem 0;'>🎵 Audio Features</h3>", unsafe_allow_html=True)
        
        # If preset selected, use those values
        if preset_selected:
            p = presets[preset_selected]
            danceability = st.slider("💃 Danceability", 0.0, 1.0, p['dance'], 0.01)
            energy = st.slider("⚡ Energy", 0.0, 1.0, p['energy'], 0.01)
            loudness = st.slider("🔊 Loudness (dB)", -15.0, 5.0, p['loud'], 0.5)
            speechiness = st.slider("🗣️ Speechiness", 0.0, 1.0, p['speech'], 0.01)
            acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, p['acoust'], 0.01)
            instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, p['instrum'], 0.01)
            liveness = st.slider("🎤 Liveness", 0.0, 1.0, p['live'], 0.01)
            valence = st.slider("😊 Valence (mood)", 0.0, 1.0, p['valence'], 0.01)
            tempo = st.slider("🎼 Tempo (BPM)", 50, 200, p['tempo'], 5)
        else:
            danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.7, 0.01)
            energy = st.slider("⚡ Energy", 0.0, 1.0, 0.7, 0.01)
            loudness = st.slider("🔊 Loudness (dB)", -15.0, 5.0, -5.0, 0.5)
            speechiness = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.05, 0.01)
            acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.1, 0.01)
            instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.0, 0.01)
            liveness = st.slider("🎤 Liveness", 0.0, 1.0, 0.1, 0.01)
            valence = st.slider("😊 Valence (mood)", 0.0, 1.0, 0.5, 0.01)
            tempo = st.slider("🎼 Tempo (BPM)", 50, 200, 120, 5)
        
        st.markdown("---")
        
        explicit = st.selectbox("🅴 Explicit Content", ["Clean", "Explicit"])
        explicit_val = 1 if explicit == "Explicit" else 0
        
        duration = st.slider("⏱️ Duration (seconds)", 60, 600, 240, 10)
        duration_ms = duration * 1000
        
        song_title = st.text_input("🎵 Song Title (for sharing)", "My Awesome Song")
        artist_name = st.text_input("🎤 Artist Name", "Your Name")
    
    with col_right:
        st.markdown("<h3 style='margin: 1.5rem 0;'>📊 Live Preview</h3>", unsafe_allow_html=True)
        
        # Live data visualization
        features_data = {
            'Feature': ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness'],
            'Value': [danceability, energy, valence, acousticness, instrumentalness]
        }
        
        fig = px.bar(features_data, x='Feature', y='Value',
                    color='Value',
                    color_continuous_scale=['#EC4899', '#7C3AED', '#06B6D4'],
                    range_y=[0, 1])
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0.1)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"),
            height=400,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Info Cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 1.5rem;'>⏱️</div>
                <div class='metric-label'>Duration</div>
                <div style='font-size: 1.3rem; color: white; margin-top: 0.5rem;'>{duration}s</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 1.5rem;'>🎼</div>
                <div class='metric-label'>Tempo</div>
                <div style='font-size: 1.3rem; color: white; margin-top: 0.5rem;'>{tempo} BPM</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Predict Button - Full Width
    st.markdown("---")
    
    if st.button("✨ PREDICT POPULARITY NOW", use_container_width=True, key="predict_main"):
        with st.spinner("🔮 Analyzing your song..."):
            # Prepare input
            input_data = pd.DataFrame({
                'danceability': [danceability],
                'energy': [energy],
                'loudness': [loudness],
                'speechiness': [speechiness],
                'acousticness': [acousticness],
                'instrumentalness': [instrumentalness],
                'liveness': [liveness],
                'valence': [valence],
                'tempo': [tempo],
                'explicit': [explicit_val],
                'duration_ms': [duration_ms]
            })
            
            # Scale features
            input_scaled = scaler.transform(input_data)
            
            # Make predictions
            pred_lr = model_lr.predict(input_scaled)[0]
            pred_rf = model_rf.predict(input_scaled)[0]
            
            # Ensure predictions are within 0-100
            pred_lr = max(0, min(100, pred_lr))
            pred_rf = max(0, min(100, pred_rf))
            
            # Average prediction
            avg_pred = (pred_lr + pred_rf) / 2
            
            # Calculate confidence
            confidence = 100 - abs(pred_lr - pred_rf) * 2
            confidence = max(50, min(100, confidence))
            
            # Store prediction
            st.session_state.current_prediction = {
                'score_lr': pred_lr,
                'score_rf': pred_rf,
                'avg_score': avg_pred,
                'confidence': confidence,
                'song_title': song_title,
                'artist_name': artist_name,
                'timestamp': datetime.now()
            }
            st.session_state.user_data['predictions_made'] += 1
        
        # Display Results
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; margin: 2rem 0;'>🎉 Your Prediction Results</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1.5, 1])
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 3rem; color: #7C3AED;'>{get_mood_emoji(pred_rf)}</div>
                <div class='metric-value' style='margin: 1rem 0;'>{int(pred_rf)}</div>
                <div class='metric-label'>Random Forest</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Score Ring Animation
            st.markdown(f"""
            <div class='glass-card' style='text-align: center; padding: 2rem;'>
                <div style='font-size: 1.2rem; margin-bottom: 1rem;'>Final Score</div>
                <div style='font-size: 3.5rem; font-weight: 700; background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{int(avg_pred)}</div>
                <div style='font-size: 1rem; color: rgba(255,255,255,0.6); margin-top: 1rem;'>out of 100</div>
                <div style='margin-top: 1.5rem; padding: 0.75rem; background: rgba(34,197,94,0.2); border-radius: 12px; color: #22C55E;'>✅ {int(confidence)}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 3rem; color: #EC4899;'>{get_mood_emoji(pred_lr)}</div>
                <div class='metric-value' style='margin: 1rem 0;'>{int(pred_lr)}</div>
                <div class='metric-label'>Linear Reg</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Insights
        st.markdown("<h3 style='margin: 1.5rem 0;'>💡 AI Insights</h3>", unsafe_allow_html=True)
        
        insights = []
        if pred_rf >= 80:
            insights.append("🔥 This could be a chart-topper!")
        elif pred_rf >= 70:
            insights.append("👍 Strong potential for success")
        elif pred_rf >= 60:
            insights.append("😐 Moderate popularity expected")
        else:
            insights.append("📈 Could improve with tweaks")
        
        if danceability > 0.7:
            insights.append("💃 High danceability appeals to clubs")
        if energy > 0.8:
            insights.append("⚡ High energy keeps listeners engaged")
        if valence > 0.7:
            insights.append("😊 Uplifting mood boosts replay rate")
        if acousticness > 0.5:
            insights.append("🎸 Acoustic elements add authenticity")
        
        insight_cols = st.columns(len(insights))
        for col, insight in zip(insight_cols, insights):
            with col:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center; padding: 1rem;'>
                    {insight}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Share Section
        st.markdown("<h3 style='margin: 1.5rem 0;'>📱 Share Your Prediction</h3>", unsafe_allow_html=True)
        
        share_col1, share_col2, share_col3, share_col4 = st.columns(4)
        
        with share_col1:
            if st.button("📸 Share on Instagram", use_container_width=True, key="share_insta"):
                # Generate share image
                share_img = generate_share_image(
                    avg_pred,
                    song_title,
                    artist_name,
                    insights
                )
                
                # Convert to bytes
                img_bytes = io.BytesIO()
                share_img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                st.success("✅ Image generated! Download and share to Instagram Stories")
                st.image(share_img, use_column_width=True)
                st.download_button(
                    label="⬇️ Download Image",
                    data=img_bytes.getvalue(),
                    file_name=f"{song_title}_prediction.png",
                    mime="image/png"
                )
        
        with share_col2:
            if st.button("𝕏 Share on Twitter", use_container_width=True, key="share_twitter"):
                tweet_text = f"🎵 Just predicted! '{song_title}' by {artist_name} has a {int(avg_pred)}% chance to HIT! 🚀 #SpotifyPredictor #MusicTok"
                twitter_url = f"https://twitter.com/intent/tweet?text={tweet_text.replace(' ', '%20')}"
                st.markdown(f"[Tweet this!]({twitter_url})", unsafe_allow_html=True)
                st.success("✅ Ready to tweet!")
        
        with share_col3:
            if st.button("✉️ Share via Email", use_container_width=True, key="share_email"):
                email_text = f"""
                I just predicted my song's popularity!
                
                Song: {song_title}
                Artist: {artist_name}
                Predicted Score: {int(avg_pred)}/100
                Confidence: {int(confidence)}%
                
                Check out Spotify Popularity Predictor:
                https://spotify-predictor.streamlit.app
                """
                mailto_link = f"mailto:?subject=Check%20out%20my%20music%20prediction&body={email_text.replace(chr(10), '%0A').replace(' ', '%20')}"
                st.markdown(f"[Send Email]({mailto_link})", unsafe_allow_html=True)
                st.success("✅ Email ready!")
        
        with share_col4:
            if st.button("💾 Save Prediction", use_container_width=True, key="save_pred"):
                st.session_state.user_data['favorites'].append({
                    'song': song_title,
                    'artist': artist_name,
                    'score': int(avg_pred),
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.success("✅ Saved to favorites!")

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

elif page == "📊 Analytics":
    st.markdown("""
    <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>
        <span class='gradient-text'>📊 Music Analytics Dashboard</span>
    </h1>
    """, unsafe_allow_html=True)
    
    # Feature Correlations
    st.markdown("<h3 style='margin: 1.5rem 0;'>📈 Feature Correlations with Popularity</h3>", unsafe_allow_html=True)
    
    features_to_analyze = ['danceability', 'energy', 'loudness', 'speechiness',
                          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    correlations = [df[feat].corr(df['popularity']) for feat in features_to_analyze]
    corr_df = pd.DataFrame({
        'Feature': features_to_analyze,
        'Correlation': correlations
    }).sort_values('Correlation', ascending=True)
    
    fig = px.barh(corr_df, x='Correlation', y='Feature',
                  color='Correlation',
                  color_continuous_scale=['#EC4899', '#7C3AED', '#06B6D4'])
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0.1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Energy vs Popularity
    st.markdown("<h3 style='margin: 1.5rem 0;'>⚡ Energy vs Popularity</h3>", unsafe_allow_html=True)
    
    fig = create_interactive_chart(df, 'energy', 'popularity', 'Energy vs Popularity Analysis')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Danceability vs Valence
    st.markdown("<h3 style='margin: 1.5rem 0;'>💃 Danceability vs Valence (Mood)</h3>", unsafe_allow_html=True)
    
    fig = px.scatter(df, x='danceability', y='valence',
                    size='popularity', color='popularity',
                    hover_data=['track_name', 'artist_name'],
                    color_continuous_scale="Viridis")
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0.1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top Features
    st.markdown("<h3 style='margin: 1.5rem 0;'>🏆 Top Features for Popularity</h3>", unsafe_allow_html=True)
    
    top_features = corr_df.tail(5)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    for col, (_, row) in zip([col1, col2, col3, col4, col5], top_features.iterrows()):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center;'>
                <h4 style='color: #06B6D4; margin-bottom: 1rem;'>{row['Feature']}</h4>
                <div style='font-size: 2.5rem; font-weight: 700; color: {'#22C55E' if row['Correlation'] > 0 else '#F97316'};'>{row['Correlation']:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE: SONG DATABASE
# ============================================================================

elif page == "🎯 Database":
    st.markdown("""
    <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>
        <span class='gradient-text'>🎯 Song Database Explorer</span>
    </h1>
    """, unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_popularity = st.slider("Min Popularity", 0, 100, 50, key="min_pop")
    with col2:
        max_popularity = st.slider("Max Popularity", 0, 100, 100, key="max_pop")
    with col3:
        artist_search = st.text_input("Search Artist", "")
    
    # Filter data
    filtered_df = df[
        (df['popularity'] >= min_popularity) & 
        (df['popularity'] <= max_popularity)
    ]
    
    if artist_search:
        filtered_df = filtered_df[
            filtered_df['artist_name'].str.contains(artist_search, case=False)
        ]
    
    filtered_df = filtered_df.sort_values('popularity', ascending=False)
    
    st.markdown("---")
    st.markdown(f"<p style='color: rgba(255,255,255,0.7);'>📊 Found <span style='color: #06B6D4; font-weight: bold;'>{len(filtered_df)}</span> songs</p>", unsafe_allow_html=True)
    
    # Display songs in cards
    for idx, (_, row) in enumerate(filtered_df.head(20).iterrows(), 1):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class='glass-card'>
                <h4 style='color: #06B6D4; margin-bottom: 0.5rem;'>{idx}. {row['track_name']}</h4>
                <p style='color: rgba(255,255,255,0.7); margin: 0.5rem 0;'>👤 {row['artist_name']}</p>
                <div style='margin-top: 1rem;'>
                    <span class='tag'>💃 {row['danceability']:.2f}</span>
                    <span class='tag'>⚡ {row['energy']:.2f}</span>
                    <span class='tag'>😊 {row['valence']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            popularity_score = row['popularity']
            if popularity_score >= 80:
                badge_class = 'success-badge'
            elif popularity_score >= 70:
                badge_class = 'info-badge'
            else:
                badge_class = 'warning-badge'
            
            st.markdown(f"""
            <div class='tag {badge_class}' style='display: block; text-align: center; font-size: 1.2rem; padding: 1rem;'>
                {int(popularity_score)}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.button("📊 Analyze", key=f"analyze_{idx}"):
                st.session_state.current_prediction = {
                    'song_title': row['track_name'],
                    'artist_name': row['artist_name'],
                    'score': row['popularity']
                }
                st.rerun()

# ============================================================================
# PAGE: LEADERBOARD
# ============================================================================

elif page == "🏆 Leaderboard":
    st.markdown("""
    <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>
        <span class='gradient-text'>🏆 Global Leaderboard</span>
    </h1>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Accuracy", "📊 Most Predictions", "📱 Most Shares"])
    
    with tab1:
        st.markdown("<h3>Top Predictors by Accuracy</h3>", unsafe_allow_html=True)
        leaderboard_acc = st.session_state.leaderboard.sort_values('Accuracy', ascending=False)
        
        for idx, (_, row) in enumerate(leaderboard_acc.iterrows(), 1):
            col1, col2, col3, col4 = st.columns([0.5, 1.5, 1, 1])
            
            with col1:
                st.markdown(f"""<div style='font-size: 1.5rem; text-align: center;'>{'🥇' if idx==1 else '🥈' if idx==2 else '🥉' if idx==3 else f'{idx}'}</div>""", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='glass-card' style='padding: 1rem;'>
                    <strong>{row['Username']}</strong>
                    <br><small style='color: rgba(255,255,255,0.6);'>{get_accuracy_badge(row['Accuracy'])}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.metric("Accuracy", f"{row['Accuracy']}%")
            
            with col4:
                st.metric("Predictions", int(row['Predictions']))
    
    with tab2:
        st.markdown("<h3>Most Active Predictors</h3>", unsafe_allow_html=True)
        leaderboard_pred = st.session_state.leaderboard.sort_values('Predictions', ascending=False)
        
        fig = px.bar(leaderboard_pred, x='Username', y='Predictions',
                    color='Predictions',
                    color_continuous_scale=['#EC4899', '#7C3AED', '#06B6D4'])
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0.1)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("<h3>Most Viral Predictors</h3>", unsafe_allow_html=True)
        leaderboard_shares = st.session_state.leaderboard.sort_values('Shares', ascending=False)
        
        for idx, (_, row) in enumerate(leaderboard_shares.iterrows(), 1):
            st.markdown(f"""
            <div class='glass-card'>
                <h4>{idx}. {row['Username']}</h4>
                <p style='margin: 0.5rem 0;'>📱 Shares: <strong>{int(row['Shares'])}</strong></p>
                <p style='margin: 0; color: rgba(255,255,255,0.6);'>Predictions: {int(row['Predictions'])}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE: PROFILE
# ============================================================================

elif page == "👤 Profile":
    st.markdown("""
    <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>
        <span class='gradient-text'>👤 Your Profile</span>
    </h1>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class='glass-card' style='text-align: center; padding: 2rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>👤</div>
            <h3>User Stats</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        col1_inner, col2_inner, col3_inner = st.columns(3)
        
        with col1_inner:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{st.session_state.user_data['predictions_made']}</div>
                <div class='metric-label'>Predictions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2_inner:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{st.session_state.user_data['accuracy_score']}</div>
                <div class='metric-label'>Accuracy %</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3_inner:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{len(st.session_state.user_data['favorites'])}</div>
                <div class='metric-label'>Favorites</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Badges
    st.markdown("<h3 style='margin: 1.5rem 0;'>🏅 Achievements</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    badges = [
        ("🚀 First Steps", "Made your first prediction"),
        ("💪 Dedicated", "10 predictions completed"),
        ("🌟 Influencer", "10 shares accomplished"),
        ("📈 Trending", "Top 10 leaderboard"),
    ]
    
    for col, (badge, desc) in zip([col1, col2, col3, col4], badges):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center; padding: 1.5rem;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{badge.split()[0]}</div>
                <p style='font-size: 0.85rem; color: rgba(255,255,255,0.6);'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Saved Predictions
    st.markdown("<h3 style='margin: 1.5rem 0;'>💾 Saved Predictions</h3>", unsafe_allow_html=True)
    
    if st.session_state.user_data['favorites']:
        for idx, fav in enumerate(st.session_state.user_data['favorites'], 1):
            st.markdown(f"""
            <div class='glass-card'>
                <h4>{idx}. {fav['song']}</h4>
                <p style='color: rgba(255,255,255,0.6); margin: 0.5rem 0;'>🎤 {fav['artist']}</p>
                <p style='margin: 0; color: #06B6D4; font-weight: bold;'>Score: {fav['score']}/100 • {fav['date']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📌 No saved predictions yet. Start predicting!")
    
    st.markdown("---")
    
    # Settings
    st.markdown("<h3 style='margin: 1.5rem 0;'>⚙️ Settings</h3>", unsafe_allow_html=True)
    
    username = st.text_input("Username", st.session_state.user_data['username'])
    if username != st.session_state.user_data['username']:
        st.session_state.user_data['username'] = username
        st.success("✅ Username updated!")
    
    if st.button("🔄 Reset All Data", use_container_width=True):
        st.session_state.user_data = {
            'username': 'Guest User',
            'predictions_made': 0,
            'accuracy_score': 0,
            'favorites': [],
            'badges': []
        }
        st.success("✅ Data reset!")
        st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🎵 Spotify Popularity Predictor v2.0")
with footer_col2:
    st.caption("Built with ❤️ using Streamlit")
with footer_col3:
    st.caption("© 2024 | Gen-Z Aesthetic Edition")
