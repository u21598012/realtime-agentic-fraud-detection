import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
import pickle
#stored csv
fd_ = pd.read_csv("AIML Dataset.csv")

print(fd_.head())
print(fd_.columns)
print(fd_['step'].unique)

def clean_data(df):
    missing_values = df.isna().sum()
    print("Missing values before cleaning:")
    print(missing_values)
    
    clean_df = df.copy()
    
    for col in clean_df.columns:
        if clean_df[col].dtype.kind in 'ifc':  
            if clean_df[col].isna().any():
                clean_df[col].fillna(clean_df[col].mean(), inplace=True)
        else:
            if clean_df[col].isna().any():
                mode_value = clean_df[col].mode()[0]
                clean_df[col].fillna(mode_value, inplace=True)
    
    le = LabelEncoder()
    clean_df['type_encoded'] = le.fit_transform(clean_df['type'])

    # if 'type' in clean_df.columns:
    #     type_dummies = pd.get_dummies(clean_df['type'], prefix='type')
    #     clean_df = pd.concat([clean_df, type_dummies], axis=1)
    

    # if 'type' in clean_df.columns:
    #     type_mapping = {'CASH_OUT': 0, 'PAYMENT': 1, 'CASH_IN': 2, 'TRANSFER': 3, 'DEBIT': 4}
    #     clean_df['type_encoded'] = clean_df['type'].map(type_mapping)
    
    # Scale numeric features
    # scaler = StandardScaler()
    # numeric_columns = [col for col in clean_df.columns if 
    #                   clean_df[col].dtype.kind in 'ifc' and 
    #                   not col.startswith('type_') and
    #                   col not in ['isfraud', 'type_encoded']]
    
    # Create scaled versions of numeric columns
    # for col in numeric_columns:
    #     clean_df[f"{col}_scaled"] = scaler.fit_transform(clean_df[[col]])
    
    # Convert column names to lowercase and replace spaces with underscores
    clean_df.columns = [col.lower().replace(' ', '_') for col in clean_df.columns]
    
    # Check missing values after cleaning
    # missing_after = clean_df.isna().sum()
    # print("\nMissing values after cleaning:")
    # print(missing_after)
    
    return clean_df, le

fd, le = clean_data(fd_)

# Transaction Behavior Features
fd['balancechangeorig'] = fd['oldbalanceorg'] - fd['newbalanceorig']
fd['balancechangedest'] = fd['newbalancedest'] - fd['oldbalancedest']
fd['errorbalanceorig'] = fd['oldbalanceorg'] - fd['amount'] - fd['newbalanceorig']
fd['errorbalancedest'] = fd['oldbalancedest'] + fd['amount'] - fd['newbalancedest']

# Interaction Features
fd['issameuser'] = (fd['nameorig'] == fd['namedest'])

print(fd.head())

# Frequency & Pattern Features
pf = pd.DataFrame()

transactions_per_user = fd.groupby('nameorig').size().reset_index(name='transactionsperuser')
pf = pd.merge(fd[['nameorig', 'namedest']], transactions_per_user, on='nameorig', how='left')

# 2. Fraud ratio per user
if 'isfraud' in fd.columns:
    # Count total transactions and fraudulent transactions per user
    fraud_counts = fd.groupby('nameorig')['isfraud'].agg(['sum', 'count']).reset_index()
    fraud_counts['fraudratioperuser'] = fraud_counts['sum'] / fraud_counts['count']
    pf = pd.merge(pf, 
                               fraud_counts[['nameorig', 'fraudratioperuser']], 
                               on='nameorig', 
                               how='left')
pf = pf[['nameorig', 'transactionsperuser'] + 
                                   (['fraudratioperuser'] if 'isfraud' in fd.columns else [])]
pf = pf.drop_duplicates('nameorig')

# pf.to_csv("fraud_pf.csv", index=False)

# print("Pattern features created and saved to 'fraud_pf.csv'")
# print(pf.head())

# Merge pattern features
fd = pd.merge(fd, pf, on='nameorig', how='left')

fd = fd.sort_values('step')

# Slice into training and testing sets, 80/20 using ID
train_size = int(0.8 * len(fd))
train_df = fd.iloc[:train_size]
test_df = fd.iloc[train_size:]

# fd.to_csv("resulting_fraud_detection_data.csv", index=False)
train_df.to_csv("train_fraud_detection_data.csv", index=False)
test_df.to_csv("test_fraud_detection_data.csv", index=False)

# user_transaction_count = {}
# user_fraud_count = {}

# live_transactions_per_user = []
# live_fraud_ratio_per_user = []

# for idx, row in fd.iterrows():
#     user = row['nameorig']
    
#     transactions = user_transaction_count.get(user, 0)
#     frauds = user_fraud_count.get(user, 0)
    
#     live_transactions_per_user.append(transactions)
#     live_fraud_ratio_per_user.append(frauds / transactions if transactions > 0 else 0)
    
#     user_transaction_count[user] = transactions + 1
#     if row['isfraud'] == 1:
#         user_fraud_count[user] = frauds + 1

# fd['livetransactionsperuser'] = live_transactions_per_user
# fd['livefraudratioperuser'] = live_fraud_ratio_per_user

# feature_cols = [
#     'step', 'amount', 'oldbalanceorg', 'newbalanceorig',
#     'oldbalancedest', 'newbalancedest',
#     'balancechangeorig', 'balancechangedest',
#     'errorbalanceorig', 'errorbalancedest',
#     'issameuser', 'livetransactionsperuser', 'livefraudratioperuser',
#     'type_encoded'
# ]

# X = fd[feature_cols]
# y = fd['isfraud']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )

# model = xgb.XGBClassifier(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     scale_pos_weight= (y == 0).sum() / (y == 1).sum(), 
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric='logloss'
# )

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)[:, 1]

# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

# model.save_model('fraud_detection_xgb_model.json')

# with open('label_encoder.pkl', 'wb') as f:
#     pickle.dump(le, f)

# print("\nModel and LabelEncoder saved successfully!")

# FUNCTION FOR LIVE PREDICTION 
def generate_features(transaction, user_stats, label_encoder):
    features = {}
    
    # Basic transaction features
    features['step'] = transaction['step']
    features['amount'] = transaction['amount']
    features['oldbalanceorg'] = transaction['oldbalanceorg']
    features['newbalanceorig'] = transaction['newbalanceorig']
    features['oldbalancedest'] = transaction['oldbalancedest']
    features['newbalancedest'] = transaction['newbalancedest']
    
    # Engineered features
    features['balancechangeorig'] = transaction['oldbalanceorg'] - transaction['newbalanceorig']
    features['balancechangeDest'] = transaction['newbalancedest'] - transaction['oldbalancedest']
    features['errorcalanceOrig'] = transaction['oldbalanceorg'] - transaction['amount'] - transaction['newbalanceorig']
    features['errorbalanceDest'] = transaction['oldbalancedest'] + transaction['amount'] - transaction['newbalancedest']
    features['issameuser'] = int(transaction['nameorig'] == transaction['namedest'])
    
    # Real-time features
    user = transaction['nameorig']
    transactions = user_stats.get(user, {}).get('transactions', 0)
    frauds = user_stats.get(user, {}).get('frauds', 0)
    
    features['livetransactionsperuser'] = transactions
    features['livefraudratioperuser'] = frauds / transactions if transactions > 0 else 0

    # Encode 'type'
    features['type_encoded'] = label_encoder.transform([transaction['type']])[0]
    
    return pd.DataFrame([features])

# model_loaded = xgb.XGBClassifier()
# model_loaded.load_model('fraud_detection_xgb_model.json')

# with open('label_encoder.pkl', 'rb') as f:
#     le_loaded = pickle.load(f)

# # Example new transaction
# new_transaction = {
#     'step': 300, 'amount': 10000,
#     'oldbalanceorg': 50000, 'newbalanceorig': 40000,
#     'oldbalancedest': 20000, 'newbalancedest': 30000,
#     'nameorig': 'C12345', 'namedest': 'C67890',
#     'type': 'TRANSFER'
# }

# # Assume live user stats tracking dictionary
# live_user_stats = {
#     'C12345': {'transactions': 5, 'frauds': 1}
# }

# # Generate features and predict
# new_features = generate_features(new_transaction, live_user_stats, le_loaded)
# prediction = model_loaded.predict(new_features)
# print("\nPrediction for new transaction (0=Not Fraud, 1=Fraud):", prediction[0])
