
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the same dataset used by Random Forest
df = pd.read_csv(
    r"archive (1)\DataCoSupplyChainDataset.csv",
    encoding="latin1"
)

numeric = [
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

categorical = [
    "Type",
    "Category Name",
    "Customer Segment",
    "Market",
    "Order Region",
    "Shipping Mode",
    "Department Name"
]

features = numeric + categorical
X = df[features]
y = df["Late_delivery_risk"].astype(int)

# Same split used by the Random Forest model
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load the saved Random Forest model
rf = joblib.load("supply_chain_risk_model_v2.pkl")

# Prepare data for Histogram-based Gradient Boosting
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]), categorical)
])

hgb = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    ))
])

print("Training Histogram-based Gradient Boosting...")
hgb.fit(X_train, y_train)

models = [
    ("Random Forest", rf),
    ("Histogram Gradient Boosting", hgb)
]

print("\nRESULTS ON THE SAME TEST SET")
print("=" * 55)

for name, model in models:
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 4))
    print("F1-score :", round(f1_score(y_test, pred, zero_division=0), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, prob), 4))

# Save the comparison model separately
joblib.dump(hgb, "supply_chain_hgb_v1.pkl")
print("\nBoosting model saved as supply_chain_hgb_v1.pkl")
print("Original models were not overwritten.")
