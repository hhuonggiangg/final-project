# 🤖 PART 2: MODEL TRAINING & EVALUATION
# Spotify Song Popularity Prediction
# Run in Jupyter: 02_Model_Training.ipynb

# ============================================================================
# CELL 1: Import Libraries
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Libraries imported!")


# ============================================================================
# CELL 2: Load Cleaned Data
# ============================================================================

df = pd.read_csv('../data/spotify_cleaned.csv')

print(f"📊 Data loaded!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())


# ============================================================================
# CELL 3: Prepare Features & Target
# ============================================================================

print("\n" + "="*70)
print("FEATURE & TARGET PREPARATION")
print("="*70)

# Define features
feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 
                   'tempo', 'explicit', 'duration_ms']

X = df[feature_columns].copy()
y = df['popularity'].copy()

print(f"\n🎯 Target Variable: popularity")
print(f"   Shape: {y.shape}")
print(f"   Mean: {y.mean():.2f}")
print(f"   Std: {y.std():.2f}")
print(f"   Range: {y.min()}-{y.max()}")

print(f"\n✨ Features Selected: {len(feature_columns)}")
print(feature_columns)

print(f"\n📊 Feature Statistics:")
print(X.describe().round(3))


# ============================================================================
# CELL 4: Check for Missing Values
# ============================================================================

print("\n" + "="*70)
print("DATA QUALITY CHECK")
print("="*70)

print("\n❓ Missing Values:")
print(f"   Features: {X.isnull().sum().sum()}")
print(f"   Target: {y.isnull().sum()}")

if X.isnull().sum().sum() > 0:
    print("\n⚠️  Handling missing values...")
    X.fillna(X.mean(), inplace=True)
    print("✅ Missing values filled with mean")


# ============================================================================
# CELL 5: Train-Test Split
# ============================================================================

print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Split Results:")
print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

print(f"\n📈 Target Distribution:")
print(f"   Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
print(f"   Test  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")


# ============================================================================
# CELL 6: Feature Scaling
# ============================================================================

print("\n" + "="*70)
print("FEATURE SCALING (StandardScaler)")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier inspection
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)

print(f"\n✨ Features scaled using StandardScaler")
print(f"\nScaled Training Data Statistics:")
print(X_train_scaled.describe().round(3))

print("\n✅ All features now have mean≈0 and std≈1")


# ============================================================================
# CELL 7: Train Linear Regression Model
# ============================================================================

print("\n" + "="*70)
print("MODEL 1: LINEAR REGRESSION")
print("="*70)

model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

print(f"\n✅ Linear Regression Model Trained!")
print(f"   Intercept: {model_lr.intercept_:.4f}")
print(f"   Coefficients: {len(model_lr.coef_)}")


# ============================================================================
# CELL 8: Train Random Forest Model
# ============================================================================

print("\n" + "="*70)
print("MODEL 2: RANDOM FOREST (Bonus!)")
print("="*70)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train_scaled, y_train)

print(f"\n✅ Random Forest Model Trained!")
print(f"   Number of trees: 100")
print(f"   Max depth: auto")


# ============================================================================
# CELL 9: Make Predictions
# ============================================================================

print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

# Linear Regression
y_train_pred_lr = model_lr.predict(X_train_scaled)
y_test_pred_lr = model_lr.predict(X_test_scaled)

# Random Forest
y_train_pred_rf = model_rf.predict(X_train_scaled)
y_test_pred_rf = model_rf.predict(X_test_scaled)

print(f"\n✅ Predictions generated!")
print(f"   Training predictions: {len(y_train_pred_lr)}")
print(f"   Test predictions: {len(y_test_pred_lr)}")

# Show sample predictions
print(f"\n📊 Sample Predictions (Linear Regression):")
for i in range(5):
    print(f"   Actual: {y_test.iloc[i]:.1f} | Predicted: {y_test_pred_lr[i]:.1f} | Error: {abs(y_test.iloc[i] - y_test_pred_lr[i]):.2f}")


# ============================================================================
# CELL 10: Evaluate Linear Regression
# ============================================================================

print("\n" + "="*70)
print("EVALUATION: LINEAR REGRESSION")
print("="*70)

# Training metrics
train_mae_lr = mean_absolute_error(y_train, y_train_pred_lr)
train_rmse_lr = np.sqrt(mean_squared_error(y_train, y_train_pred_lr))
train_r2_lr = r2_score(y_train, y_train_pred_lr)

# Test metrics
test_mae_lr = mean_absolute_error(y_test, y_test_pred_lr)
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
test_r2_lr = r2_score(y_test, y_test_pred_lr)

print(f"\n🎯 TRAINING METRICS (Linear Regression):")
print(f"   R² Score: {train_r2_lr:.4f} ({train_r2_lr*100:.1f}% accuracy)")
print(f"   MAE: {train_mae_lr:.4f} (avg error in points)")
print(f"   RMSE: {train_rmse_lr:.4f}")

print(f"\n🎯 TEST METRICS (Linear Regression):")
print(f"   R² Score: {test_r2_lr:.4f} ({test_r2_lr*100:.1f}% accuracy)")
print(f"   MAE: {test_mae_lr:.4f} (avg error in points)")
print(f"   RMSE: {test_rmse_lr:.4f}")

print(f"\n💡 Interpretation:")
print(f"   The model explains {test_r2_lr*100:.1f}% of popularity variance")
print(f"   Average prediction error: ±{test_mae_lr:.2f} points (out of 100)")


# ============================================================================
# CELL 11: Evaluate Random Forest
# ============================================================================

print("\n" + "="*70)
print("EVALUATION: RANDOM FOREST")
print("="*70)

# Training metrics
train_mae_rf = mean_absolute_error(y_train, y_train_pred_rf)
train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
train_r2_rf = r2_score(y_train, y_train_pred_rf)

# Test metrics
test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)
test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
test_r2_rf = r2_score(y_test, y_test_pred_rf)

print(f"\n🎯 TRAINING METRICS (Random Forest):")
print(f"   R² Score: {train_r2_rf:.4f} ({train_r2_rf*100:.1f}% accuracy)")
print(f"   MAE: {train_mae_rf:.4f}")
print(f"   RMSE: {train_rmse_rf:.4f}")

print(f"\n🎯 TEST METRICS (Random Forest):")
print(f"   R² Score: {test_r2_rf:.4f} ({test_r2_rf*100:.1f}% accuracy)")
print(f"   MAE: {test_mae_rf:.4f}")
print(f"   RMSE: {test_rmse_rf:.4f}")


# ============================================================================
# CELL 12: Model Comparison
# ============================================================================

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Linear Regression': {
        'Test R²': test_r2_lr,
        'Test MAE': test_mae_lr,
        'Test RMSE': test_rmse_lr
    },
    'Random Forest': {
        'Test R²': test_r2_rf,
        'Test MAE': test_mae_rf,
        'Test RMSE': test_rmse_rf
    }
}).T

print("\n📊 Performance Comparison:")
print(comparison_df.round(4))

# Determine best model
if test_r2_lr > test_r2_rf:
    print("\n✅ WINNER: Linear Regression (simpler & better generalization)")
    best_model = model_lr
    best_predictions = y_test_pred_lr
    best_r2 = test_r2_lr
    best_mae = test_mae_lr
else:
    print("\n✅ WINNER: Random Forest (more complex & captures non-linearity)")
    best_model = model_rf
    best_predictions = y_test_pred_rf
    best_r2 = test_r2_rf
    best_mae = test_mae_rf


# ============================================================================
# CELL 13: Visualize Predictions
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Linear Regression
axes[0].scatter(y_test, y_test_pred_lr, alpha=0.6, s=50, color='#1DB954')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Popularity', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Popularity', fontsize=12, fontweight='bold')
axes[0].set_title(f'Linear Regression (R²={test_r2_lr:.3f})', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Random Forest
axes[1].scatter(y_test, y_test_pred_rf, alpha=0.6, s=50, color='#FF1744')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Popularity', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Popularity', fontsize=12, fontweight='bold')
axes[1].set_title(f'Random Forest (R²={test_r2_rf:.3f})', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/09_predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 14: Residual Analysis
# ============================================================================

train_residuals_lr = y_train - y_train_pred_lr
test_residuals_lr = y_test - y_test_pred_lr

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residuals distribution
axes[0, 0].hist(train_residuals_lr, bins=20, color='#1DB954', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Training Residuals Distribution', fontsize=12, fontweight='bold')
axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].grid(axis='y', alpha=0.3)

axes[0, 1].hist(test_residuals_lr, bins=20, color='#FF1744', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Test Residuals Distribution', fontsize=12, fontweight='bold')
axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].grid(axis='y', alpha=0.3)

# Residuals vs Predicted
axes[1, 0].scatter(y_train_pred_lr, train_residuals_lr, alpha=0.6, color='#1DB954')
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Popularity', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Training Residuals Plot', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(y_test_pred_lr, test_residuals_lr, alpha=0.6, color='#FF1744')
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Predicted Popularity', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Test Residuals Plot', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/10_residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 15: Feature Importance (Linear Regression)
# ============================================================================

print("\n" + "="*70)
print("FEATURE IMPORTANCE (Linear Regression)")
print("="*70)

feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model_lr.coef_,
    'Abs_Coefficient': np.abs(model_lr.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\n📊 Feature Coefficients (Impact on Popularity):")
print(feature_importance.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Positive coefficients
top_positive = feature_importance.nlargest(8, 'Coefficient')
axes[0].barh(range(len(top_positive)), top_positive['Coefficient'].values, color='#1DB954')
axes[0].set_yticks(range(len(top_positive)))
axes[0].set_yticklabels(top_positive['Feature'].values)
axes[0].set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
axes[0].set_title('Top Features Increasing Popularity', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Negative coefficients
top_negative = feature_importance.nsmallest(8, 'Coefficient')
axes[1].barh(range(len(top_negative)), top_negative['Coefficient'].values, color='#FF1744')
axes[1].set_yticks(range(len(top_negative)))
axes[1].set_yticklabels(top_negative['Feature'].values)
axes[1].set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
axes[1].set_title('Top Features Decreasing Popularity', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/11_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 16: Feature Importance (Random Forest)
# ============================================================================

print("\n" + "="*70)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*70)

rf_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Random Forest Feature Importance:")
print(rf_importance.to_string(index=False))

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
rf_importance.plot(x='Feature', y='Importance', kind='barh', ax=ax, color='#1DB954', legend=False)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Random Forest Feature Importance', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/12_rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 17: Save Models
# ============================================================================

print("\n" + "="*70)
print("SAVING MODELS & ARTIFACTS")
print("="*70)

import os
os.makedirs('../models', exist_ok=True)

# Save Linear Regression Model
with open('../models/model_lr.pkl', 'wb') as f:
    pickle.dump(model_lr, f)
print("✅ Linear Regression model saved: models/model_lr.pkl")

# Save Random Forest Model
with open('../models/model_rf.pkl', 'wb') as f:
    pickle.dump(model_rf, f)
print("✅ Random Forest model saved: models/model_rf.pkl")

# Save Scaler
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved: models/scaler.pkl")

# Save Feature Names
with open('../models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("✅ Feature names saved: models/feature_names.pkl")

# Save Model Info
model_info = {
    'model_type_lr': 'LinearRegression',
    'model_type_rf': 'RandomForestRegressor',
    'features': feature_columns,
    'target': 'popularity',
    'test_r2_lr': test_r2_lr,
    'test_mae_lr': test_mae_lr,
    'test_rmse_lr': test_rmse_lr,
    'test_r2_rf': test_r2_rf,
    'test_mae_rf': test_mae_rf,
    'test_rmse_rf': test_rmse_rf,
    'best_model': 'linear_regression' if test_r2_lr > test_r2_rf else 'random_forest'
}

with open('../models/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✅ Model info saved: models/model_info.pkl")


# ============================================================================
# CELL 18: Error Analysis
# ============================================================================

print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70)

test_results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted_LR': y_test_pred_lr,
    'Predicted_RF': y_test_pred_rf,
    'Error_LR': np.abs(y_test.values - y_test_pred_lr),
    'Error_RF': np.abs(y_test.values - y_test_pred_rf),
})

print("\n📊 Error Statistics (Linear Regression):")
print(f"   Min Error: {test_results['Error_LR'].min():.2f}")
print(f"   Max Error: {test_results['Error_LR'].max():.2f}")
print(f"   Mean Error: {test_results['Error_LR'].mean():.2f}")
print(f"   Median Error: {test_results['Error_LR'].median():.2f}")

print("\n🎯 Prediction Accuracy (Linear Regression):")
within_5 = (test_results['Error_LR'] <= 5).sum()
within_10 = (test_results['Error_LR'] <= 10).sum()
print(f"   Within ±5 points: {within_5} ({within_5/len(test_results)*100:.1f}%)")
print(f"   Within ±10 points: {within_10} ({within_10/len(test_results)*100:.1f}%)")


# ============================================================================
# CELL 19: Final Summary
# ============================================================================

print("\n" + "="*70)
print("PART 2 SUMMARY")
print("="*70)

print(f"""
✅ MODEL TRAINING COMPLETE!

📊 DATASET:
   • Total samples: {len(df)}
   • Training samples: {len(X_train)}
   • Test samples: {len(X_test)}
   • Features: {len(feature_columns)}

🤖 MODELS TRAINED:
   1. Linear Regression
   2. Random Forest (bonus!)

📈 LINEAR REGRESSION PERFORMANCE:
   • Test R² Score: {test_r2_lr:.4f} ({test_r2_lr*100:.1f}%)
   • Test MAE: {test_mae_lr:.4f} points
   • Test RMSE: {test_rmse_lr:.4f}
   
   ➜ Model explains {test_r2_lr*100:.1f}% of popularity variance
   ➜ Average prediction error: ±{test_mae_lr:.2f} points

📈 RANDOM FOREST PERFORMANCE:
   • Test R² Score: {test_r2_rf:.4f} ({test_r2_rf*100:.1f}%)
   • Test MAE: {test_mae_rf:.4f} points
   • Test RMSE: {test_rmse_rf:.4f}

💾 SAVED ARTIFACTS:
   ✓ models/model_lr.pkl
   ✓ models/model_rf.pkl
   ✓ models/scaler.pkl
   ✓ models/feature_names.pkl
   ✓ models/model_info.pkl

📊 GENERATED VISUALIZATIONS:
   ✓ results/09_predictions_comparison.png
   ✓ results/10_residuals_analysis.png
   ✓ results/11_feature_importance.png
   ✓ results/12_rf_feature_importance.png

🎯 TOP PREDICTIVE FEATURES:
   1. Energy: {feature_importance.iloc[0]['Coefficient']:.4f}
   2. Danceability: {feature_importance.iloc[1]['Coefficient']:.4f}
   3. Valence: {feature_importance.iloc[2]['Coefficient']:.4f}

👉 Next: Part 3 - Deploy Streamlit Webapp!
""")

print("✨ Part 2 (Model Training) COMPLETE!")
