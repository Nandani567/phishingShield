import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

df = pd.read_csv('phishing_dataset.csv')
print("Total missing values:", df.isnull().sum().sum())



y = df["Result"].replace({-1: 0, 1: 1})  # phishing = 1, legitimate = 0
X = df.drop("Result", axis=1)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    num_leaves=31,
    objective='binary',
    random_state=42
)
lgbm_model.fit(X_train, y_train)
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]



xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

ensemble_probs = (lgbm_probs + xgb_probs) / 2

# Adjust threshold for high recall
threshold = 0.46
ensemble_pred = (ensemble_probs >= threshold).astype(int)


# Catch rare phishing sites missed by ML
final_pred = ensemble_pred.copy()
final_pred[(X_test['having_IP_Address']==1) & (X_test['URL_Length']<=0)] = 1


print("\n---Ensemble + Rule Evaluation---")
print("Accuracy:", accuracy_score(y_test, final_pred))
print("Precision:", precision_score(y_test, final_pred))
print("Recall:", recall_score(y_test, final_pred))
print("F1-score:", f1_score(y_test, final_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, final_pred))

# Remaining false negatives
X_test_copy = X_test.copy()
X_test_copy['y_true'] = y_test
X_test_copy['y_pred'] = final_pred
false_negatives = X_test_copy[(X_test_copy['y_true']==1) & (X_test_copy['y_pred']==0)]
print("\nRemaining missed phishing sites:", len(false_negatives))
print(false_negatives.head())


X_test_copy.to_csv('phishing_test_results_final.csv', index=False)
joblib.dump(lgbm_model, "phishing_model.pkl")