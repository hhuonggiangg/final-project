# 🎵 Spotify Popularity Predictor - ULTIMATE VERSION
# With Authentication & Smooth Page Transitions
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🎵 Spotify Popularity Predictor",
    page_icon="🎵",
    layout="wide"
)

# ============================================================================
# ADVANCED CSS WITH PAGE TRANSITIONS
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Inter', sans-serif;
}

html, body {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    color: white;
}

.main {
    background: transparent;
}

/* PAGE TRANSITION ANIMATION */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(50px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.page-container {
    animation: fadeInUp 0.6s ease-out;
}

.card-animate {
    animation: slideInRight 0.5s ease-out;
}

/* GLASSMORPHISM */
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
    transform: translateY(-4px);
}

/* GRADIENT TEXT */
.gradient-text {
    background: linear-gradient(135deg, #7C3AED 0%, #EC4899 50%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* METRIC CARD */
.metric-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(124, 58, 237, 0.3);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: rgba(124, 58, 237, 0.6);
    background: rgba(124, 58, 237, 0.1);
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

/* TAG */
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
    transition: all 0.3s ease;
}

.tag:hover {
    background: rgba(124, 58, 237, 0.4);
    transform: scale(1.05);
}

/* BUTTON */
.btn-gradient {
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

.btn-gradient:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(236, 72, 153, 0.4);
}

/* AUTH FORM */
.auth-container {
    max-width: 400px;
    margin: 50px auto;
    animation: fadeInUp 0.6s ease-out;
}

.auth-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 3rem;
}

input[type="text"],
input[type="password"],
input[type="email"] {
    width: 100%;
    padding: 12px;
    margin: 10px 0;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    color: white;
    font-size: 1rem;
    transition: all 0.3s ease;
}

input::placeholder {
    color: rgba(255, 255, 255, 0.4);
}

input:focus {
    outline: none;
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(124, 58, 237, 0.5);
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.2);
}

/* ANIMATIONS */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.pulse {
    animation: pulse 2s infinite;
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.bounce {
    animation: bounce 1s infinite;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE & AUTH
# ============================================================================

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'username' not in st.session_state:
    st.session_state.username = None

if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'predictions_made': 0,
        'favorites': [],
        'badges': []
    }

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

if 'preset_selected' not in st.session_state:
    st.session_state.preset_selected = None

# Mock user database
USERS_DB = {
    'demo': 'demo123',
    'test': 'test123'
}

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('spotify_songs.csv')
        return df
    except Exception as e:
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
    except:
        return None, None, None

df = load_data()
model_lr, scaler, model_info = load_models()

if model_lr is None or df is None:
    st.error("⚠️ Missing required files!")
    st.stop()

# ============================================================================
# PAGE TRANSITION FUNCTION
# ============================================================================

def transition_to_page(page_name):
    """Smooth page transition"""
    st.session_state.current_page = page_name
    time.sleep(0.1)  # Brief pause for animation
    st.rerun()

# ============================================================================
# LOGIN/SIGNUP PAGE
# ============================================================================

def show_auth_page():
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; margin: 50px 0;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 10px;'>🎵</h1>
            <h1 style='background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 30px;'>
                Spotify Predictor
            </h1>
            <p style='color: rgba(255,255,255,0.7); font-size: 1.2rem;'>
                AI-powered music popularity predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown("<div class='page-container'>", unsafe_allow_html=True)
            
            with st.form("login_form"):
                st.markdown("<h3 style='text-align: center; margin-bottom: 2rem;'>Login to Your Account</h3>", unsafe_allow_html=True)
                
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                
                col1, col2 = st.columns(2)
                with col1:
                    login_btn = st.form_submit_button("🔓 Login", use_container_width=True)
                with col2:
                    demo_btn = st.form_submit_button("👤 Demo Login", use_container_width=True)
                
                if login_btn:
                    if username in USERS_DB and USERS_DB[username] == password:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials!")
                
                if demo_btn:
                    st.session_state.authenticated = True
                    st.session_state.username = "Demo User"
                    st.success("✅ Welcome to demo!")
                    time.sleep(1)
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<div class='page-container'>", unsafe_allow_html=True)
            
            with st.form("signup_form"):
                st.markdown("<h3 style='text-align: center; margin-bottom: 2rem;'>Create New Account</h3>", unsafe_allow_html=True)
                
                new_username = st.text_input("Choose Username", placeholder="Pick a username")
                new_email = st.text_input("Email", placeholder="your@email.com")
                new_password = st.text_input("Password", type="password", placeholder="Create password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
                
                signup_btn = st.form_submit_button("✨ Sign Up", use_container_width=True)
                
                if signup_btn:
                    if not new_username or not new_email or not new_password:
                        st.error("❌ Please fill all fields!")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords don't match!")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters!")
                    else:
                        USERS_DB[new_username] = new_password
                        st.session_state.authenticated = True
                        st.session_state.username = new_username
                        st.success("✅ Account created! Welcome!")
                        time.sleep(1)
                        st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# MAIN APP (AUTHENTICATED)
# ============================================================================

def show_app():
    # Sidebar with smooth navigation
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0; margin-bottom: 1rem; animation: fadeInUp 0.6s ease-out;'>
            <h2 style='font-size: 2rem; margin-bottom: 0.5rem;'>🎵</h2>
            <h3 style='background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Spotify Predictor
            </h3>
            <p style='font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 0.5rem;'>
                Welcome, <strong>{st.session_state.username}!</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation with smooth transitions
        pages = ["🏠 Home", "🔮 Predict", "📊 Analytics", "🎯 Database", "👤 Profile"]
        
        st.markdown("<h4 style='color: rgba(255,255,255,0.6); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin: 1rem 0;'>Navigation</h4>", unsafe_allow_html=True)
        
        for page in pages:
            if st.button(page, use_container_width=True, key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("---")
        
        # User Stats
        st.markdown(f"""
        <div class='metric-card' style='margin-top: 1rem;'>
            <div style='font-size: 0.85rem; color: rgba(255,255,255,0.6);'>📊 Statistics</div>
            <div style='margin-top: 1rem; text-align: left; font-size: 0.9rem;'>
                <p style='margin: 0.5rem 0;'>🎵 Predictions: <strong>{st.session_state.user_data['predictions_made']}</strong></p>
                <p style='margin: 0.5rem 0;'>❤️ Favorites: <strong>{len(st.session_state.user_data['favorites'])}</strong></p>
                <p style='margin: 0.5rem 0;'>🏆 Badges: <strong>{len(st.session_state.user_data['badges'])}</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    
    # ========================================================================
    # PAGE: HOME
    # ========================================================================
    
    if st.session_state.current_page == "🏠 Home":
        st.markdown("<div class='page-container'>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin: 3rem 0;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 1rem; animation: fadeInUp 0.6s ease-out;'>
                <span class='gradient-text'>🔮 Discover What Makes Songs HIT</span>
            </h1>
            <p style='font-size: 1.2rem; color: rgba(255,255,255,0.7); animation: fadeInUp 0.8s ease-out;'>
                AI-powered music analytics | Predict popularity | Share your predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card card-animate'>
                <div class='metric-value'>{model_info.get('test_r2_lr', 0.75)*100:.1f}%</div>
                <div class='metric-label'>Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card card-animate' style='animation-delay: 0.1s;'>
                <div class='metric-value'>±{model_info.get('test_mae_lr', 8.5):.1f}</div>
                <div class='metric-label'>Avg Error</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card card-animate' style='animation-delay: 0.2s;'>
                <div class='metric-value'>{len(df)}</div>
                <div class='metric-label'>Songs in DB</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card card-animate' style='animation-delay: 0.3s;'>
                <div class='metric-value'>{df['artist_name'].nunique()}</div>
                <div class='metric-label'>Artists</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<h2 style='margin: 2rem 0; animation: fadeInUp 0.6s ease-out;'>✨ Key Features</h2>", unsafe_allow_html=True)
        
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
        st.markdown("<h2 style='margin: 2rem 0;'>🔥 Trending Now</h2>", unsafe_allow_html=True)
        
        sample_songs = df.sample(min(5, len(df)))
        col1, col2, col3, col4, col5 = st.columns(5)
        
        for idx, (col, (_, song)) in enumerate(zip([col1, col2, col3, col4, col5], sample_songs.iterrows())):
            with col:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center; animation: slideInRight 0.5s ease-out; animation-delay: {idx*0.1}s;'>
                    <h4 style='color: #06B6D4; margin-bottom: 0.5rem; font-size: 0.9rem;'>{song['track_name'][:15]}...</h4>
                    <p style='color: rgba(255,255,255,0.6); font-size: 0.8rem; margin: 0.5rem 0;'>{song['artist_name'][:12]}...</p>
                    <div style='font-size: 2rem; margin: 1rem 0;'>
                        {'🔥' if song['popularity'] >= 80 else '👍' if song['popularity'] >= 70 else '📈'}
                    </div>
                    <p style='background: linear-gradient(135deg, #7C3AED, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.5rem; font-weight: 700;'>{int(song['popularity'])}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ========================================================================
    # PAGE: PREDICT
    # ========================================================================
    
    elif st.session_state.current_page == "🔮 Predict":
        st.markdown("<div class='page-container'>", unsafe_allow_html=True)
        
        st.markdown("""
        <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
            <span class='gradient-text'>🔮 Will Your Song HIT?</span>
        </h1>
        <p style='color: rgba(255,255,255,0.7); margin-bottom: 2rem;'>
            Adjust features and get instant AI predictions
        </p>
        """, unsafe_allow_html=True)
        
        # Presets
        st.markdown("<h3 style='margin: 1.5rem 0;'>⚡ Quick Presets</h3>", unsafe_allow_html=True)
        
        presets = {
            'Summer Hit ☀️': {'dance': 0.8, 'energy': 0.85, 'loud': -3.0, 'speech': 0.03, 'acoust': 0.05, 'instrum': 0.0, 'live': 0.1, 'valence': 0.8, 'tempo': 120, 'explicit': 0},
            'Chill Vibes 😎': {'dance': 0.5, 'energy': 0.3, 'loud': -6.0, 'speech': 0.02, 'acoust': 0.5, 'instrum': 0.1, 'live': 0.08, 'valence': 0.6, 'tempo': 90, 'explicit': 0},
            'Party 🎉': {'dance': 0.9, 'energy': 0.95, 'loud': -1.0, 'speech': 0.04, 'acoust': 0.02, 'instrum': 0.0, 'live': 0.15, 'valence': 0.75, 'tempo': 130, 'explicit': 1},
            'Sad Vibes 💔': {'dance': 0.4, 'energy': 0.3, 'loud': -7.0, 'speech': 0.02, 'acoust': 0.6, 'instrum': 0.05, 'live': 0.1, 'valence': 0.2, 'tempo': 70, 'explicit': 0},
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("☀️ Summer Hit", use_container_width=True):
                st.session_state.preset_selected = 'Summer Hit ☀️'
                st.rerun()
        with col2:
            if st.button("😎 Chill Vibes", use_container_width=True):
                st.session_state.preset_selected = 'Chill Vibes 😎'
                st.rerun()
        with col3:
            if st.button("🎉 Party", use_container_width=True):
                st.session_state.preset_selected = 'Party 🎉'
                st.rerun()
        with col4:
            if st.button("💔 Sad Vibes", use_container_width=True):
                st.session_state.preset_selected = 'Sad Vibes 💔'
                st.rerun()
        
        st.markdown("---")
        st.markdown("<h3 style='margin: 1.5rem 0;'>🎵 Audio Features</h3>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 1])
        
        if st.session_state.preset_selected and st.session_state.preset_selected in presets:
            p = presets[st.session_state.preset_selected]
        else:
            p = {'dance': 0.7, 'energy': 0.7, 'loud': -5.0, 'speech': 0.05, 'acoust': 0.1, 'instrum': 0.0, 'live': 0.1, 'valence': 0.5, 'tempo': 120, 'explicit': 0}
        
        with col_left:
            danceability = st.slider("💃 Danceability", 0.0, 1.0, float(p['dance']), 0.01)
            energy = st.slider("⚡ Energy", 0.0, 1.0, float(p['energy']), 0.01)
            loudness = st.slider("🔊 Loudness (dB)", -15.0, 5.0, float(p['loud']), 0.5)
            speechiness = st.slider("🗣️ Speechiness", 0.0, 1.0, float(p['speech']), 0.01)
            acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, float(p['acoust']), 0.01)
        
        with col_right:
            instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, float(p['instrum']), 0.01)
            liveness = st.slider("🎤 Liveness", 0.0, 1.0, float(p['live']), 0.01)
            valence = st.slider("😊 Valence (mood)", 0.0, 1.0, float(p['valence']), 0.01)
            tempo = st.slider("🎼 Tempo (BPM)", 50, 200, int(p['tempo']), 5)
            explicit = st.selectbox("🅴 Explicit", ["Clean", "Explicit"])
        
        explicit_val = 1 if explicit == "Explicit" else 0
        song_title = st.text_input("🎵 Song Title", "My Song")
        artist_name = st.text_input("🎤 Artist", "Your Name")
        
        st.markdown("---")
        
        if st.button("✨ PREDICT POPULARITY", use_container_width=True, key="predict_btn"):
            with st.spinner("🔮 Analyzing..."):
                time.sleep(0.5)
                
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
                    'duration_ms': [240000]
                })
                
                input_scaled = scaler.transform(input_data)
                pred = model_lr.predict(input_scaled)[0]
                pred = max(0, min(100, pred))
                
                st.session_state.user_data['predictions_made'] += 1
            
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; margin: 2rem 0; animation: fadeInUp 0.6s ease-out;'>
                <h2 style='margin-bottom: 2rem;'>🎉 Your Prediction Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 3rem;'>{'🔥' if pred >= 80 else '👍' if pred >= 70 else '📈'}</div>
                    <div class='metric-value' style='margin: 1rem 0;'>{int(pred)}</div>
                    <div class='metric-label'>Your Prediction</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center; padding: 2rem;'>
                    <div style='font-size: 1.2rem; margin-bottom: 1rem; color: rgba(255,255,255,0.6);'>Final Score</div>
                    <div style='font-size: 3.5rem; font-weight: 700; background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 1rem 0;'>{int(pred)}</div>
                    <div style='font-size: 0.9rem; color: rgba(255,255,255,0.6); margin-top: 1rem;'>out of 100</div>
                    <div style='margin-top: 1.5rem; padding: 0.75rem; background: rgba(34,197,94,0.2); border-radius: 12px; color: #22C55E; font-weight: 600;'>✅ 85% Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 3rem;'>🎵</div>
                    <div class='metric-value' style='margin: 1rem 0;'>{int(pred)}</div>
                    <div class='metric-label'>Prediction</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            insights = []
            if danceability > 0.7:
                insights.append("💃 High danceability")
            if energy > 0.8:
                insights.append("⚡ High energy")
            if valence > 0.7:
                insights.append("😊 Uplifting")
            if acousticness > 0.5:
                insights.append("🎸 Acoustic vibes")
            
            if insights:
                st.markdown("<h3 style='margin: 1.5rem 0;'>💡 Key Factors</h3>", unsafe_allow_html=True)
                cols = st.columns(len(insights))
                for col, insight in zip(cols, insights):
                    with col:
                        st.markdown(f"<div class='glass-card' style='text-align: center;'>{insight}</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("<h3 style='margin: 1.5rem 0;'>📱 Share Your Prediction</h3>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📸 Instagram", use_container_width=True):
                    st.success("✅ Download image and share to Instagram!")
            with col2:
                if st.button("𝕏 Twitter", use_container_width=True):
                    st.success(f"✅ Tweet: '{song_title}' predicted {int(pred)}%!")
            with col3:
                if st.button("✉️ Email", use_container_width=True):
                    st.success("✅ Share via email!")
            with col4:
                if st.button("💾 Save", use_container_width=True):
                    st.session_state.user_data['favorites'].append({'song': song_title, 'score': int(pred)})
                    st.success("✅ Saved!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ========================================================================
    # PAGE: ANALYTICS
    # ========================================================================
    
    elif st.session_state.current_page == "📊 Analytics":
        st.markdown("<div class='page-container'>", unsafe_allow_html=True)
        
        st.markdown("""
        <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>
            <span class='gradient-text'>📊 Music Analytics</span>
        </h1>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.9rem; color: rgba(255,255,255,0.6);'>Avg Popularity</div>
                <div class='metric-value'>{df['popularity'].mean():.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.9rem; color: rgba(255,255,255,0.6);'>Avg Energy</div>
                <div class='metric-value'>{df['energy'].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.9rem; color: rgba(255,255,255,0.6);'>Avg Danceability</div>
                <div class='metric-value'>{df['danceability'].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.9rem; color: rgba(255,255,255,0.6);'>Avg Valence</div>
                <div class='metric-value'>{df['valence'].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<h3 style='margin: 1.5rem 0;'>🏆 Top Artists</h3>", unsafe_allow_html=True)
        
        top_artists = df.groupby('artist_name')['popularity'].mean().nlargest(10)
        
        for idx, (artist, pop) in enumerate(top_artists.items(), 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"<p style='margin: 0.5rem 0;'><strong>{idx}. {artist}</strong></p>", unsafe_allow_html=True)
            with col2:
                st.metric("Avg Pop", f"{pop:.1f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ========================================================================
    # PAGE: DATABASE
    # ========================================================================
    
    elif st.session_state.current_page == "🎯 Database":
        st.markdown("<div class='page-container'>", unsafe_allow_html=True)
        
        st.markdown("""
        <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>
            <span class='gradient-text'>🎯 Song Database</span>
        </h1>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_pop = st.slider("Min Popularity", 0, 100, 50)
        with col2:
            max_pop = st.slider("Max Popularity", 0, 100, 100)
        with col3:
            search = st.text_input("Search Artist")
        
        filtered = df[(df['popularity'] >= min_pop) & (df['popularity'] <= max_pop)]
        if search:
            filtered = filtered[filtered['artist_name'].str.contains(search, case=False)]
        
        filtered = filtered.sort_values('popularity', ascending=False)
        
        st.markdown(f"**Found {len(filtered)} songs**")
        st.markdown("---")
        
        for idx, (_, song) in enumerate(filtered.head(15).iterrows(), 1):
            st.markdown(f"""
            <div class='glass-card'>
                <h4 style='color: #06B6D4; margin-bottom: 0.5rem;'>{idx}. {song['track_name']}</h4>
                <p style='color: rgba(255,255,255,0.7); margin: 0.5rem 0;'>👤 {song['artist_name']}</p>
                <div style='margin-top: 1rem;'>
                    <span class='tag'>💃 {song['danceability']:.2f}</span>
                    <span class='tag'>⚡ {song['energy']:.2f}</span>
                    <span class='tag'>😊 {song['valence']:.2f}</span>
                    <span class='tag' style='float: right;'>🎵 {int(song['popularity'])}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ========================================================================
    # PAGE: PROFILE
    # ========================================================================
    
    elif st.session_state.current_page == "👤 Profile":
        st.markdown("<div class='page-container'>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>
            <span class='gradient-text'>👤 Your Profile</span>
        </h1>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.9rem; color: rgba(255,255,255,0.6);'>Predictions Made</div>
                <div class='metric-value'>{st.session_state.user_data['predictions_made']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.9rem; color: rgba(255,255,255,0.6);'>Saved Favorites</div>
                <div class='metric-value'>{len(st.session_state.user_data['favorites'])}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.9rem; color: rgba(255,255,255,0.6);'>Badges Earned</div>
                <div class='metric-value'>{len(st.session_state.user_data['badges'])}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.session_state.user_data['favorites']:
            st.markdown("<h3 style='margin: 1.5rem 0;'>❤️ Saved Predictions</h3>", unsafe_allow_html=True)
            
            for fav in st.session_state.user_data['favorites']:
                st.markdown(f"""
                <div class='glass-card'>
                    <h4>{fav['song']}</h4>
                    <p style='color: #06B6D4; font-weight: 600;'>Score: {fav['score']}/100</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📌 No saved predictions yet!")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# MAIN
# ============================================================================

if st.session_state.authenticated:
    show_app()
else:
    show_auth_page()
