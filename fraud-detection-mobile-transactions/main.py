import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump

# ── 1. Load Dataset ──
df = pd.read_csv('Fraud_Analysis_Dataset(in).csv')
print(f"Dataset Shape: {df.shape}")

# ── 2. Feature Engineering ──
# Create balance difference features (same as notebook)
df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

# One-Hot Encode the transaction type column (drop_first=True, same as notebook)
type_dummies = pd.get_dummies(df['type'], prefix='type', drop_first=True)
print(f"Encoded type columns: {list(type_dummies.columns)}")

# ── 3. Select Features ──
numeric_features = [
    'step', 'amount',
    'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balance_diff_orig', 'balance_diff_dest'
]

X = pd.concat([df[numeric_features], type_dummies], axis=1)
y = df['isFraud']

print(f"Features used: {list(X.columns)}")

# ── 4. Train / Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ── 5. Train XGBoost Model ──
model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# ── 6. Evaluate ──
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print(f"\nROC-AUC Score: {roc_auc_score(y_test, prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=['Legitimate', 'Fraud']))

# ── 7. Save the Model ──
dump(model, 'fraud_model.joblib')
dump(list(X.columns), 'feature_names.joblib')
print("\n✅ Model saved. Run: streamlit run app.py")