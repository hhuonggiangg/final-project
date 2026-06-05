# 🌐 PART 3: STREAMLIT WEBAPP
# Spotify Song Popularity Prediction
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🎵 Spotify Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/spotify_songs.csv')
        return df
    except:
        st.error("❌ Could not load data. Make sure data/spotify_songs.csv exists!")
        return None

@st.cache_resource
def load_models():
    try:
        with open('models/model_lr.pkl', 'rb') as f:
            model_lr = pickle.load(f)
        with open('models/model_rf.pkl', 'rb') as f:
            model_rf = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model_lr, model_rf, scaler, feature_names, model_info
    except:
        return None, None, None, None, None
        # Load from file (nếu file tồn tại)
try:
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except:
    # Nếu file không tồn tại, hardcode
    feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 
                     'tempo', 'explicit', 'duration_ms']

# Load data
df = load_data()

# Try to load models
model_lr, model_rf, scaler, feature_names, model_info = load_models()

# Check if models are loaded
if model_lr is None:
    st.warning("⚠️ Models not found! Run Part 1 & 2 first to train models.")
    st.info("Steps:\n1. Run: jupyter notebook notebooks/01_EDA.ipynb\n2. Run: jupyter notebook notebooks/02_Model_Training.ipynb\n3. Then run this app again!")
    st.stop()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🎵 Spotify Popularity Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Home", "🔮 Predict Popularity", "📊 Analytics", "🎯 Song Database", "ℹ️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About This App:**
AI-powered music analytics tool that predicts 
song popularity based on audio features!

**Data:** 100 Spotify songs
**Model:** Linear Regression & Random Forest
""")

# ============================================================================
# PAGE 1: HOME
# ============================================================================

if page == "🏠 Home":
    st.title("🎵 Spotify Song Popularity Predictor")
    st.markdown("*AI-powered analytics for your favorite songs!*")
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy", f"{model_info['test_r2_lr']*100:.1f}%", "R² Score")
    with col2:
        st.metric("Avg Error", f"±{model_info['test_mae_lr']:.1f}", "points")
    with col3:
        st.metric("Songs", len(df), "in database")
    with col4:
        st.metric("Artists", df['artist_name'].nunique(), "unique")
    
    st.markdown("---")
    
    # Features
    st.header("✨ Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🔮 Predict Popularity**\nPredicts song popularity score (0-100)")
    with col2:
        st.info("**📊 Analytics**\nExplore music trends and patterns")
    with col3:
        st.info("**🎯 Song Database**\nBrowse and search songs")
    
    st.markdown("---")
    
    # Quick Stats
    st.header("📈 Database Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Popularity", f"{df['popularity'].mean():.1f}", "out of 100")
    with col2:
        st.metric("Avg Energy", f"{df['energy'].mean():.2f}", "out of 1.0")
    with col3:
        st.metric("Avg Danceability", f"{df['danceability'].mean():.2f}", "out of 1.0")
    with col4:
        st.metric("Avg Valence", f"{df['valence'].mean():.2f}", "mood (positive)")
    
    st.markdown("---")
    
    # Top Artists
    st.header("🏆 Top Artists")
    top_artists = df.groupby('artist_name')['popularity'].mean().nlargest(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_artists)))
    ax.barh(range(len(top_artists)), top_artists.values, color=colors)
    ax.set_yticks(range(len(top_artists)))
    ax.set_yticklabels(top_artists.index)
    ax.set_xlabel('Average Popularity', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Artists by Average Popularity', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    
    for i, v in enumerate(top_artists.values):
        ax.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig, use_container_width=True)


# ============================================================================
# PAGE 2: PREDICT POPULARITY
# ============================================================================

elif page == "🔮 Predict Popularity":
    st.title("🔮 Predict Song Popularity")
    st.markdown("*Enter song audio features to predict popularity*")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎵 Audio Features")
        
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
    
    with col1:
        st.markdown("")
        predict_btn = st.button("🎵 PREDICT POPULARITY", use_container_width=True, key="predict")
    
    st.markdown("---")
    
    if predict_btn:
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
        pred_avg = (pred_lr + pred_rf) / 2
        
        st.header("✨ Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Linear Regression", f"{pred_lr:.1f}", "popularity")
        with col2:
            st.metric("Random Forest", f"{pred_rf:.1f}", "popularity")
        with col3:
            st.metric("Average", f"{pred_avg:.1f}", "popularity")
        
        st.markdown("---")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge
            fig, ax = plt.subplots(figsize=(8, 6))
            
            popularity_pct = pred_avg / 100
            colors_list = ['#FF1744', '#FFA726', '#FFD54F', '#AED581', '#1DB954']
            
            wedges, texts = ax.pie([popularity_pct, 1-popularity_pct],
                                   colors=['#1DB954', '#e0e0e0'],
                                   startangle=90,
                                   wedgeprops=dict(width=0.3))
            
            ax.text(0, 0, f'{pred_avg:.0f}', ha='center', va='center', 
                   fontsize=40, fontweight='bold', color='#1DB954')
            ax.text(0, -0.3, 'Popularity Score', ha='center', fontsize=12)
            
            st.pyplot(fig)
        
        with col2:
            # Interpretation
            st.write("### 📊 Interpretation")
            
            if pred_avg >= 80:
                st.success("🌟 **HITS POTENTIAL!** This song is likely to be very popular!")
            elif pred_avg >= 70:
                st.info("⭐ **STRONG!** Good potential for popularity!")
            elif pred_avg >= 60:
                st.info("👍 **GOOD!** Solid song with decent potential!")
            elif pred_avg >= 50:
                st.warning("⚠️ **FAIR!** May have moderate popularity!")
            else:
                st.error("❌ **LOW!** May struggle to gain popularity!")
            
            st.markdown("---")
            
            # Feature contribution
            st.write("### 🎯 Key Characteristics")
            
            characteristics = []
            if energy > 0.7:
                characteristics.append("✓ High Energy - Energetic vibes")
            if danceability > 0.7:
                characteristics.append("✓ Highly Danceable - Great for dancing")
            if valence > 0.7:
                characteristics.append("✓ Positive Mood - Uplifting song")
            if acousticness > 0.5:
                characteristics.append("✓ Acoustic - Organic sound")
            if tempo > 120:
                characteristics.append("✓ Fast Tempo - Upbeat rhythm")
            
            if characteristics:
                for char in characteristics:
                    st.write(char)
            else:
                st.write("- Moderate characteristics")


# ============================================================================
# PAGE 3: ANALYTICS
# ============================================================================

elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")
    st.markdown("*Explore Spotify music trends and patterns*")
    st.markdown("---")
    
    # Overview
    st.header("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Songs", len(df))
    with col2:
        st.metric("Avg Popularity", f"{df['popularity'].mean():.1f}")
    with col3:
        st.metric("Unique Artists", df['artist_name'].nunique())
    with col4:
        st.metric("Avg Tempo", f"{df['tempo'].mean():.0f} BPM")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎵 Popularity Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['popularity'], bins=15, color='#1DB954', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Popularity Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Songs', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("⚡ Energy Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['energy'], bins=15, color='#FF1744', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Energy Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Songs', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature Correlations
    st.subheader("🔗 Feature Correlations with Popularity")
    
    features_to_analyze = ['danceability', 'energy', 'loudness', 'speechiness',
                          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    correlations = [df[feat].corr(df['popularity']) for feat in features_to_analyze]
    corr_df = pd.DataFrame({
        'Feature': features_to_analyze,
        'Correlation': correlations
    }).sort_values('Correlation', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1DB954' if x > 0 else '#FF1744' for x in corr_df['Correlation']]
    ax.barh(range(len(corr_df)), corr_df['Correlation'].values, color=colors)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df['Feature'].values)
    ax.set_xlabel('Correlation with Popularity', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Energy vs Popularity
    st.subheader("⚡ Energy vs Popularity")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df['energy'], df['popularity'], alpha=0.6, s=100, color='#1DB954')
    
    # Trendline
    z = np.polyfit(df['energy'], df['popularity'], 1)
    p = np.poly1d(z)
    ax.plot(sorted(df['energy']), p(sorted(df['energy'])), "r--", linewidth=2, label='Trend')
    
    ax.set_xlabel('Energy', fontsize=11, fontweight='bold')
    ax.set_ylabel('Popularity', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Danceability vs Valence
    st.subheader("💃 Danceability vs Valence (Mood)")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['danceability'], df['valence'], 
                        c=df['popularity'], cmap='RdYlGn', s=100, alpha=0.6)
    ax.set_xlabel('Danceability', fontsize=11, fontweight='bold')
    ax.set_ylabel('Valence (Mood)', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Popularity', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)


# ============================================================================
# PAGE 4: SONG DATABASE
# ============================================================================

elif page == "🎯 Song Database":
    st.title("🎯 Song Database")
    st.markdown("*Browse and search the Spotify songs database*")
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_popularity = st.slider("Min Popularity", 0, 100, 50)
    with col2:
        max_popularity = st.slider("Max Popularity", 0, 100, 100)
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
    
    st.subheader(f"📊 Found {len(filtered_df)} songs")
    
    # Display songs
    for idx, (_, row) in enumerate(filtered_df.head(20).iterrows(), 1):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.write(f"""
            **{idx}. {row['track_name']}**
            - Artist: {row['artist_name']}
            - Popularity: {'⭐'*int(row['popularity']/20)} {row['popularity']:.0f}/100
            - Energy: {row['energy']:.2f} | Danceability: {row['danceability']:.2f} | Valence: {row['valence']:.2f}
            """)
        
        with col2:
            if row['popularity'] >= 80:
                st.success(f"{row['popularity']:.0f}")
            elif row['popularity'] >= 70:
                st.info(f"{row['popularity']:.0f}")
            else:
                st.warning(f"{row['popularity']:.0f}")
        
        st.divider()


# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    
    st.header("📚 Project Overview")
    st.write("""
    This is a **complete End-to-End Data Science Project** that demonstrates:
    
    1. **Part 1: Exploratory Data Analysis (EDA)**
       - Load Spotify dataset
       - Analyze audio features
       - Visualize trends and patterns
       - Data quality checks
    
    2. **Part 2: Model Training**
       - Feature scaling
       - Train Linear Regression & Random Forest
       - Model evaluation and comparison
       - Feature importance analysis
    
    3. **Part 3: Web Deployment**
       - Interactive Streamlit webapp
       - Real-time predictions
       - Analytics dashboard
       - Song database explorer
    """)
    
    st.markdown("---")
    
    st.header("🛠️ Technologies Used")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("""
        **Data Science**
        - Pandas
        - NumPy
        - Scikit-learn
        """)
    with col2:
        st.write("""
        **Visualization**
        - Matplotlib
        - Seaborn
        """)
    with col3:
        st.write("""
        **Web Framework**
        - Streamlit
        - Python 3.8+
        """)
    
    st.markdown("---")
    
    st.header("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Linear Regression R²", f"{model_info['test_r2_lr']:.1%}")
    with col2:
        st.metric("Random Forest R²", f"{model_info['test_r2_rf']:.1%}")
    with col3:
        st.metric("Avg Error (LR)", f"±{model_info['test_mae_lr']:.1f}")
    
    st.markdown("---")
    
    st.header("🎵 Audio Features Explained")
    st.write("""
    - **Danceability**: How suitable a track is for dancing (0-1)
    - **Energy**: Intensity and activity of the track (0-1)
    - **Loudness**: Overall loudness in decibels (dB)
    - **Speechiness**: Presence of spoken words (0-1)
    - **Acousticness**: Likelihood of being acoustic (0-1)
    - **Instrumentalness**: Lack of vocals (0-1)
    - **Liveness**: Presence of audience (0-1)
    - **Valence**: Musical positiveness/happiness (0-1)
    - **Tempo**: Speed in beats per minute (BPM)
    """)
    
    st.markdown("---")
    
    st.header("📈 Learning Outcomes")
    st.write("""
    By building this project, you'll learn:
    
    ✅ **Data Science Fundamentals**
    - Exploratory data analysis
    - Feature engineering
    - Model training and evaluation
    - Cross-validation and hyperparameter tuning
    
    ✅ **Machine Learning**
    - Linear regression concepts
    - Tree-based models
    - Model comparison and selection
    - Performance metrics (R², MAE, RMSE)
    
    ✅ **Web Development**
    - Streamlit framework basics
    - Interactive UI components
    - Data visualization in web apps
    - Deployment best practices
    
    ✅ **Software Engineering**
    - Code organization
    - Model serialization (pickle)
    - Error handling
    - Documentation
    """)
    
    st.markdown("---")
    st.success("🎉 Built with ❤️ using Python, Scikit-learn & Streamlit!")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🎵 Spotify Popularity Predictor")
with col2:
    st.caption("Made with Streamlit")
with col3:
    st.caption("© 2024 | End-to-End Data Science Project")
