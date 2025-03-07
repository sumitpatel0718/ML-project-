import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# 1. Create Synthetic Dataset
# -----------------------------
np.random.seed(42)
n_samples = 200

# Generate continuous features (angles and speed)
elbow = np.random.uniform(20, 160, n_samples)
knee = np.random.uniform(20, 160, n_samples)
shoulder = np.random.uniform(20, 160, n_samples)
back = np.random.uniform(20, 160, n_samples)
speed = np.random.uniform(80, 150, n_samples)

# Generate categorical features (encoded as integers)
bowling_type = np.random.choice([0, 1], n_samples)      # e.g., 0: spin, 1: fast
action = np.random.choice([0, 1], n_samples)              # e.g., 0: underarm, 1: overarm
pitch_type = np.random.choice([0, 1, 2], n_samples)       # e.g., 0: grass, 1: concrete, 2: artificial

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
# Define "true" weights for generating the risk (for simulation purposes)
true_weights = np.array([0.25, 0.25, 0.15, 0.15, 0.1, 0.05, 0.05, 0.1])
# Compute risk score (linear combination) and add noise
risk_true = df[['elbow', 'knee', 'shoulder', 'back', 'speed', 'bowling_type', 'action', 'pitch_type']].values.dot(true_weights)
risk_true += np.random.normal(0, 5, n_samples)  # add some noise
# Binarize risk score: 1 if above median, else 0
threshold_true = np.median(risk_true)
df['injury_risk'] = (risk_true > threshold_true).astype(int)

# -----------------------------
# 3. Prepare Train-Test Split
# -----------------------------
X = df[['elbow', 'knee', 'shoulder', 'back', 'speed', 'bowling_type', 'action', 'pitch_type']]
y = df['injury_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 4. Define Candidate Weight Sets
# -----------------------------
# Start with default equal weights
default_weights = np.array([1/8] * 8)
candidate_weights_list = []
weight_descriptions = []

# Create 12 candidate sets by perturbing one weight at a time (cyclically)
for i in range(12):
    candidate = default_weights.copy()
    idx = i % 8  # choose feature index to change
    # Increase the weight of feature 'idx' by 0.05 (or 0.1 for second cycle) 
    increment = 0.05 if i < 8 else 0.1
    candidate[idx] += increment
    # To keep the sum equal to 1, subtract the excess equally from the remaining features.
    excess = candidate[idx] - default_weights[idx]
    for j in range(8):
        if j != idx:
            candidate[j] -= excess / 7
    # Append candidate weight vector and record description
    candidate_weights_list.append(candidate)
    weight_descriptions.append(f"Candidate {i+1} (change feature {idx})")

# -----------------------------
# 5. Evaluate Each Candidate Weight Set
# -----------------------------
results = []
for desc, weights in zip(weight_descriptions, candidate_weights_list):
    # Compute risk scores on training set
    risk_train = X_train.values.dot(weights)
    # Determine threshold from training set (using median)
    threshold_candidate = np.median(risk_train)
    
    # Compute risk scores on test set and predict labels
    risk_test = X_test.values.dot(weights)
    y_pred = (risk_test > threshold_candidate).astype(int)
    
    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)
    results.append({'Candidate': desc, 'Weights': np.round(weights, 3), 'Accuracy': np.round(acc, 3)})

results_df = pd.DataFrame(results)
print("Candidate Weights and Corresponding Accuracy:")
print(results_df)

# -----------------------------
# 6. Find the Best Candidate Weight Set
# -----------------------------
best_candidate = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nBest Candidate Weight Set:")
print(best_candidate)
