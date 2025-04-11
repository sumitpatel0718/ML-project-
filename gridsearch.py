import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import itertools

warnings.filterwarnings("ignore")

# -----------------------------
# 1. Generate Synthetic Data
# -----------------------------
np.random.seed(42)
n_samples = 200

elbow = np.random.uniform(20, 160, n_samples)
knee = np.random.uniform(20, 160, n_samples)
shoulder = np.random.uniform(20, 160, n_samples)
back = np.random.uniform(20, 160, n_samples)
speed = np.random.uniform(80, 150, n_samples)

bowling_type = np.random.choice([0, 1], n_samples)  # 0: spin, 1: fast
action = np.random.choice([0, 1], n_samples)
pitch_type = np.random.choice([0, 1, 2], n_samples)

# Create DataFrame
df = pd.DataFrame({
    'elbow': elbow,
    'knee': knee,
    'shoulder': shoulder,
    'back': back,
    'speed': speed,
    'bowling_type': bowling_type,
    'action': action,
    'pitch_type': pitch_type
})

# -----------------------------
# 2. Generate True Risk and Labels
# -----------------------------
true_weights = np.array([0.25, 0.25, 0.15, 0.15, 0.1, 0.05, 0.05, 0.1])
risk_true = df[['elbow', 'knee', 'shoulder', 'back', 'speed', 'bowling_type', 'action', 'pitch_type']].values.dot(true_weights)
risk_true += np.random.normal(0, 5, n_samples)
threshold_true = np.median(risk_true)
df['injury_risk'] = (risk_true > threshold_true).astype(int)

# -----------------------------
# 3. Prepare Train-Test Split
# -----------------------------
X = df[['elbow', 'knee', 'shoulder', 'back', 'speed', 'bowling_type', 'action', 'pitch_type']]
y = df['injury_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 4. Grid Search for Optimal Weights
# -----------------------------
print("Running grid search...")

# Define a coarse grid of possible weight values
weight_values = np.arange(0, 1.1, 0.25)  # Step size 0.25 for fast iteration

results = []

# Generate all possible combinations of 8 weights that sum approximately to 1
for weights in itertools.product(weight_values, repeat=8):
    weights = np.array(weights)
    if np.isclose(np.sum(weights), 1.0):
        # Compute risk scores on training data
        risk_train = X_train.values.dot(weights)
        threshold = np.median(risk_train)
        
        # Predict on test data
        risk_test = X_test.values.dot(weights)
        y_pred = (risk_test > threshold).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Weights': np.round(weights, 3),
            'Accuracy': np.round(acc, 3)
        })

# -----------------------------
# 5. Display Results
# -----------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

print("\nTop 5 Weight Vectors by Accuracy:")
print(results_df.head())

print("\nBest Weight Vector:")
print(results_df.iloc[0])
