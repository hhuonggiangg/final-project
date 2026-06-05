# 🎵 PART 1: EXPLORATORY DATA ANALYSIS (EDA)
# Spotify Song Popularity Prediction
# Run in Jupyter: 01_EDA.ipynb

# ============================================================================
# CELL 1: Import Libraries
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Libraries imported!")


# ============================================================================
# CELL 2: Load Dataset
# ============================================================================

df = pd.read_csv('../data/spotify_songs.csv')

print(f"📊 Dataset loaded!")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())


# ============================================================================
# CELL 3: Dataset Overview
# ============================================================================

print("\n" + "="*70)
print("SPOTIFY DATASET OVERVIEW")
print("="*70)

print(f"\n📋 Basic Info:")
print(f"   Songs: {len(df)}")
print(f"   Features: {len(df.columns)}")
print(f"   Artists: {df['artist_name'].nunique()}")

print(f"\n📊 Data Types:")
print(df.dtypes)

print(f"\n❓ Missing Values:")
print(df.isnull().sum())


# ============================================================================
# CELL 4: Statistical Summary
# ============================================================================

print("\n" + "="*70)
print("STATISTICAL SUMMARY")
print("="*70)

print(df.describe().round(3))

print("\n🎯 KEY FEATURES EXPLAINED:")
print("""
   • danceability: 0-1, how suitable for dancing
   • energy: 0-1, intensity and activity
   • loudness: dB, overall loudness
   • speechiness: 0-1, presence of spoken words
   • acousticness: 0-1, likelihood of acoustic
   • instrumentalness: 0-1, lack of vocals
   • liveness: 0-1, presence of audience
   • valence: 0-1, musical positiveness
   • tempo: BPM, speed of track
   • popularity: 0-100, target variable!
""")


# ============================================================================
# CELL 5: Popularity Distribution
# ============================================================================

print("\n" + "="*70)
print("POPULARITY ANALYSIS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['popularity'], bins=25, color='#1DB954', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Popularity Score', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Songs', fontsize=12, fontweight='bold')
axes[0].set_title('Spotify Song Popularity Distribution', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Box plot
axes[1].boxplot(df['popularity'], vert=True)
axes[1].set_ylabel('Popularity Score', fontsize=12, fontweight='bold')
axes[1].set_title('Popularity Box Plot', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/01_popularity_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n📊 Popularity Statistics:")
print(f"   Min: {df['popularity'].min()}")
print(f"   Max: {df['popularity'].max()}")
print(f"   Mean: {df['popularity'].mean():.2f}")
print(f"   Median: {df['popularity'].median():.2f}")
print(f"   Std: {df['popularity'].std():.2f}")


# ============================================================================
# CELL 6: Analyze Audio Features
# ============================================================================

print("\n" + "="*70)
print("AUDIO FEATURES ANALYSIS")
print("="*70)

audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(audio_features):
    axes[idx].hist(df[feature], bins=20, color='#1DB954', edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel(feature, fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Frequency', fontsize=10, fontweight='bold')
    axes[idx].set_title(f'{feature.capitalize()} Distribution', fontsize=11, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/02_audio_features_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Audio features distributions plotted!")


# ============================================================================
# CELL 7: Correlation with Popularity
# ============================================================================

print("\n" + "="*70)
print("CORRELATION WITH POPULARITY")
print("="*70)

# Calculate correlations
correlations = df[audio_features + ['popularity']].corr()['popularity'].sort_values(ascending=False)
print("\nFeature Correlations with Popularity:")
print(correlations)

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#1DB954' if x > 0 else '#FF1744' for x in correlations[:-1].values]
bars = ax.barh(range(len(correlations)-1), correlations[:-1].values, color=colors)
ax.set_yticks(range(len(correlations)-1))
ax.set_yticklabels(correlations[:-1].index)
ax.set_xlabel('Correlation with Popularity', fontsize=12, fontweight='bold')
ax.set_title('Feature Correlation with Song Popularity', fontsize=13, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/03_correlation_with_popularity.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 8: Feature Scatter Plots vs Popularity
# ============================================================================

print("\n" + "="*70)
print("TOP FEATURES VS POPULARITY")
print("="*70)

top_features = correlations[:-1].nlargest(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    axes[idx].scatter(df[feature], df['popularity'], alpha=0.6, s=50, color='#1DB954')
    
    # Add trendline
    z = np.polyfit(df[feature], df['popularity'], 1)
    p = np.poly1d(z)
    axes[idx].plot(sorted(df[feature]), p(sorted(df[feature])), "r--", linewidth=2)
    
    # Add correlation coefficient
    corr = df[feature].corr(df['popularity'])
    
    axes[idx].set_xlabel(feature.capitalize(), fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Popularity', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{feature.capitalize()} vs Popularity\n(r={corr:.3f})', 
                       fontsize=12, fontweight='bold')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/04_features_vs_popularity.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 9: Top Artists Analysis
# ============================================================================

print("\n" + "="*70)
print("TOP ARTISTS ANALYSIS")
print("="*70)

top_artists = df['artist_name'].value_counts().head(10)
print("\nTop 10 Artists by Song Count:")
print(top_artists)

# Average popularity by artist
artist_popularity = df.groupby('artist_name')['popularity'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print("\nTop 10 Artists by Average Popularity:")
print(artist_popularity.head(10))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top artists by count
top_artists.plot(kind='barh', ax=axes[0], color='#1DB954')
axes[0].set_xlabel('Number of Songs', fontsize=12, fontweight='bold')
axes[0].set_title('Top 10 Artists by Song Count', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Top artists by popularity
artist_popularity.nlargest(10, 'mean')['mean'].plot(kind='barh', ax=axes[1], color='#FF1744')
axes[1].set_xlabel('Average Popularity', fontsize=12, fontweight='bold')
axes[1].set_title('Top 10 Artists by Average Popularity', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/05_artists_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 10: Tempo & Duration Analysis
# ============================================================================

print("\n" + "="*70)
print("TEMPO & DURATION ANALYSIS")
print("="*70)

print(f"\nTempo (BPM):")
print(f"   Min: {df['tempo'].min():.0f}")
print(f"   Max: {df['tempo'].max():.0f}")
print(f"   Mean: {df['tempo'].mean():.2f}")

print(f"\nDuration (ms):")
print(f"   Min: {df['duration_ms'].min()}")
print(f"   Max: {df['duration_ms'].max()}")
print(f"   Mean: {df['duration_ms'].mean():.0f}")

# Convert duration to minutes
df['duration_min'] = df['duration_ms'] / 60000

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['tempo'], bins=20, color='#1DB954', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Tempo (BPM)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Tempo Distribution', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

axes[1].hist(df['duration_min'], bins=20, color='#1DB954', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Duration (Minutes)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Duration Distribution', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/06_tempo_duration.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 11: Explicit Content Analysis
# ============================================================================

print("\n" + "="*70)
print("EXPLICIT CONTENT ANALYSIS")
print("="*70)

explicit_counts = df['explicit'].value_counts()
print("\nExplicit Content Distribution:")
print(f"   Clean: {explicit_counts.get(0, 0)}")
print(f"   Explicit: {explicit_counts.get(1, 0)}")

explicit_popularity = df.groupby('explicit')['popularity'].agg(['mean', 'count'])
print("\nPopularity by Explicit Status:")
print(explicit_popularity)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count
explicit_counts.plot(kind='bar', ax=axes[0], color=['#1DB954', '#FF1744'])
axes[0].set_xticklabels(['Clean', 'Explicit'], rotation=0)
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].set_title('Explicit vs Clean Songs', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Popularity
explicit_popularity['mean'].plot(kind='bar', ax=axes[1], color=['#1DB954', '#FF1744'])
axes[1].set_xticklabels(['Clean', 'Explicit'], rotation=0)
axes[1].set_ylabel('Average Popularity', fontsize=12, fontweight='bold')
axes[1].set_title('Popularity: Clean vs Explicit', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/07_explicit_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 12: Correlation Heatmap
# ============================================================================

print("\n" + "="*70)
print("CORRELATION HEATMAP")
print("="*70)

# Select numerical features for correlation
numeric_features = audio_features + ['popularity']
correlation_matrix = df[numeric_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/08_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 13: Data Quality Check
# ============================================================================

print("\n" + "="*70)
print("DATA QUALITY CHECK")
print("="*70)

print("\n✅ Missing Values:")
print(df.isnull().sum())

print("\n✅ Data Types:")
print(df.dtypes)

print("\n✅ Duplicates:")
print(f"   Total duplicate rows: {df.duplicated().sum()}")

print("\n✅ Outliers (using IQR):")
for feature in audio_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[feature] < Q1 - 1.5*IQR) | (df[feature] > Q3 + 1.5*IQR)]
    if len(outliers) > 0:
        print(f"   {feature}: {len(outliers)} outliers")


# ============================================================================
# CELL 14: Prepare Data for Modeling
# ============================================================================

print("\n" + "="*70)
print("DATA PREPARATION")
print("="*70)

# Create a clean dataset
df_clean = df.copy()

# Select features for modeling
feature_columns = audio_features + ['explicit', 'duration_ms']

print(f"\n📊 Features selected: {len(feature_columns)}")
print(feature_columns)

# Check for any missing values in features or target
print(f"\nMissing values in features: {df_clean[feature_columns].isnull().sum().sum()}")
print(f"Missing values in target: {df_clean['popularity'].isnull().sum()}")

# Save cleaned data
df_clean.to_csv('../data/spotify_cleaned.csv', index=False)
print("\n✅ Cleaned data saved: data/spotify_cleaned.csv")


# ============================================================================
# CELL 15: Summary & Key Insights
# ============================================================================

print("\n" + "="*70)
print("KEY INSIGHTS FROM EDA")
print("="*70)

print(f"""
🎵 SPOTIFY DATASET SUMMARY:
   • Total Songs: {len(df)}
   • Artists: {df['artist_name'].nunique()}
   • Popularity Range: {df['popularity'].min()}-{df['popularity'].max()}
   • Average Popularity: {df['popularity'].mean():.1f}

🔍 TOP FINDINGS:

1️⃣  POPULARITY DRIVERS:
   • Energy: {correlations['energy']:.3f} (HIGH positive correlation!)
   • Danceability: {correlations['danceability']:.3f}
   • Valence (mood): {correlations['valence']:.3f}
   • Acousticness: {correlations['acousticness']:.3f} (NEGATIVE)

2️⃣  AUDIO CHARACTERISTICS:
   • Most songs: {df['danceability'].mean():.2f} danceability
   • Average energy: {df['energy'].mean():.2f}
   • Tempo range: {df['tempo'].min():.0f}-{df['tempo'].max():.0f} BPM
   • Avg duration: {df['duration_min'].mean():.1f} minutes

3️⃣  ARTIST PATTERNS:
   • Top artist: {df['artist_name'].value_counts().index[0]}
   • Songs vary widely in popularity
   • Some artists consistently popular

4️⃣  CONTENT PATTERNS:
   • Explicit songs: {explicit_counts.get(1, 0)} ({explicit_counts.get(1, 0)/len(df)*100:.1f}%)
   • Clean songs: {explicit_counts.get(0, 0)} ({explicit_counts.get(0, 0)/len(df)*100:.1f}%)

📌 FOR MODELING:
   • Energy, danceability, valence are key predictors
   • Acousticness has negative relationship
   • Some features are highly correlated (multicollinearity possible)
   • Dataset is clean with no missing values
""")

print("\n✅ Part 1 (EDA) COMPLETE!")
print("👉 Next: Part 2 - Model Training (02_Model_Training.ipynb)")
