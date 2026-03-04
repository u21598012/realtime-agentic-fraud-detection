import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
import pickle
import mlflow
mlflow.set_tracking_uri("file:./mlruns")

fd = pd.read_csv("train_fraud_detection_data.csv")
le = LabelEncoder()

feature_cols = [
    'step', 'amount', 'oldbalanceorg', 'newbalanceorig',
    'oldbalancedest', 'newbalancedest',
    'balancechangeorig', 'balancechangedest',
    'errorbalanceorig', 'errorbalancedest',
    'issameuser', 
    'type_encoded'
]

X = fd[feature_cols]
y = fd['isfraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=218, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight= (y == 0).sum() / (y == 1).sum(), 
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))


    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

    model.save_model('fraud_detection_xgb_model.json')
    mlflow.log_artifact("fraud_detection_xgb_model.json")

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("\nModel and LabelEncoder saved successfully!")

