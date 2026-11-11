
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load the DataCo dataset
df = pd.read_csv(
    r"archive (1)\DataCoSupplyChainDataset.csv",
    encoding="latin1"
)

numeric_features = [
    "Days for shipment (scheduled)",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Product Price",
    "Category Id",
    "Department Id",
    "Latitude",
    "Longitude"
]

categorical_features = [
    "Type",
    "Category Name",
    "Customer Segment",
    "Market",
    "Order Region",
    "Shipping Mode",
    "Department Name"
]

features = numeric_features + categorical_features
X = df[features]
y = df["Late_delivery_risk"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load the trained DataCo model
model = joblib.load("supply_chain_risk_model_v2.pkl")

# Majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

models = {
    "Majority baseline": baseline,
    "DataCo Random Forest": model
}

print("\nComparison on the same DataCo test records")
print("-" * 65)

for name, current_model in models.items():
    pred = current_model.predict(X_test)

    if hasattr(current_model, "predict_proba"):
        prob = current_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
    else:
        auc = float("nan")

    print(f"\n{name}")
    print("Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 4))
    print("F1-score :", round(f1_score(y_test, pred, zero_division=0), 4))
    print("ROC-AUC  :", round(auc, 4))

print("\nComparison complete.")
