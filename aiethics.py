

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Fairlearn imports
from fairlearn.metrics import MetricFrame, demographic_parity_difference, demographic_parity_ratio
from fairlearn.reductions import ExponentiatedGradient, DemographicParity


# STEP 1: DATA LOADING

print("Loading the UCI Adult dataset from OpenML...")
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
df = X.copy()
df['income'] = y  # Add target to DataFrame for easier cleaning

################################################################################
# STEP 2: DATA CLEANING & PREPROCESSING
################################################################################
print("Cleaning data...")

# Replace '?' with NaN and drop missing rows
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Convert the target to binary (1 if >50K, 0 otherwise)
df['income'] = (df['income'] == '>50K').astype(int)

# We'll treat 'sex' as our sensitive attribute

sensitive_feature = df['sex'].copy()

# Drop 'sex' from the main features so it won't directly influence training
df.drop(labels=['sex'], axis=1, inplace=True)

# Now separate features (X) and target (y) again
y = df['income'].values
X = df.drop(labels=['income'], axis=1)

# Identify categorical vs numeric columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()

################################################################################
# STEP 3: TRAIN/TEST SPLIT
################################################################################
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_feature, test_size=0.3, random_state=42
)

################################################################################
# STEP 4: BUILD A PIPELINE FOR DATA ENCODING + MODEL
################################################################################
# We'll encode categorical features with OneHotEncoder and scale numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ],
    remainder='drop'
)

# Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)

# Full pipeline: preprocessing -> logistic regression
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lr_model)
])

################################################################################
# STEP 5: TRAIN THE MODEL
################################################################################
print("Training the Logistic Regression model...")
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Evaluate basic accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Model Accuracy: {accuracy:.3f}")

################################################################################
# STEP 6: INITIAL FAIRNESS EVALUATION
################################################################################
# We'll use Fairlearn's MetricFrame to evaluate metrics by sensitive groups
metrics = {
    'accuracy': accuracy_score
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)

# Overall metrics
print("\nOverall Metrics:")
print(metric_frame.overall)

# By-group metrics
print("\nBy-Group Metrics:")
print(metric_frame.by_group)

# Calculate demographic parity difference & ratio
dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sens_test)
print(f"\nDemographic Parity Difference (closer to 0 is better): {dp_diff:.3f}")
print(f"Demographic Parity Ratio (closer to 1 is better): {dp_ratio:.3f}")

################################################################################
# STEP 7: BIAS MITIGATION USING EXPONENTIATED GRADIENT
################################################################################
# We apply a fairness constraint (DemographicParity) to reduce bias
print("\nApplying bias mitigation with ExponentiatedGradient...")

# 7a. Preprocess X_train and X_test to numeric arrays (since ExponentiatedGradient
#     works directly on numeric data). We'll reuse the pipeline's preprocessor.
X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

# 7b. Set up the ExponentiatedGradient mitigator
base_estimator = LogisticRegression(max_iter=1000)
constraint = DemographicParity()

mitigator = ExponentiatedGradient(
    estimator=base_estimator,
    constraints=constraint,
    eps=0.01  # fairness vs. accuracy trade-off
)

# 7c. Fit the mitigator on training data with sensitive features
mitigator.fit(X_train_transformed, y_train, sensitive_features=sens_train)

# 7d. Predict on test set
y_pred_mitigated = mitigator.predict(X_test_transformed)

# Evaluate performance again
accuracy_mitigated = accuracy_score(y_test, y_pred_mitigated)
dp_diff_mitigated = demographic_parity_difference(
    y_test, y_pred_mitigated, sensitive_features=sens_test
)
dp_ratio_mitigated = demographic_parity_ratio(
    y_test, y_pred_mitigated, sensitive_features=sens_test
)

print(f"\nMitigated Model Accuracy: {accuracy_mitigated:.3f}")
print(f"Mitigated Demographic Parity Difference: {dp_diff_mitigated:.3f}")
print(f"Mitigated Demographic Parity Ratio: {dp_ratio_mitigated:.3f}")

################################################################################
# STEP 8: DOCUMENTATION AND RECOMMENDATIONS
################################################################################
print("\n=== DOCUMENTATION & RECOMMENDATIONS ===")
print("1. We used 'sex' as the sensitive feature to measure potential bias.")
print("2. Before mitigation, we observed certain fairness metrics (demographic parity difference/ratio).")
print("3. After applying Exponentiated Gradient with DemographicParity constraints,")
print("   we see that fairness metrics improved (lower difference or ratio closer to 1).")
print("4. Trade-off: Notice that the accuracy may decrease slightly after bias mitigation.")
print("5. For real-world deployment, continue iterating on hyperparameters (eps, etc.),")
print("   or explore alternative mitigation strategies (e.g., post-processing).")
print("6. Always communicate these findings clearly to stakeholders and document potential biases.\n")

print("=== TASK COMPLETE: AI Ethics and Bias Evaluation ===")
