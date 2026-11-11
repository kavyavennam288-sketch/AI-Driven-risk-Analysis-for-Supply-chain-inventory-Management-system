
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Load the dataset
csv_path = r"C:\Users\admin\OneDrive\Attachments\SupplyChainAI\ml-model\archive (1)\DataCoSupplyChainDataset.csv"
df = pd.read_csv(csv_path, encoding="latin1")

# Sort orders by date
df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"], errors="coerce"
)
df = df.dropna(subset=["order date (DateOrders)"])
df = df.sort_values("order date (DateOrders)").reset_index(drop=True)

# Use the same features as the trained models
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
    "Longitude",
]

categorical_features = [
    "Type",
    "Category Name",
    "Customer Segment",
    "Market",
    "Order Region",
    "Shipping Mode",
    "Department Name",
]

features = numeric_features + categorical_features
target = "Late_delivery_risk"

# Keep the newest 20% of orders for testing
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_test = test_df[features]
y_test = test_df[target]

print("Training period:")
print(train_df["order date (DateOrders)"].min(), "to",
      train_df["order date (DateOrders)"].max())

print("\nFuture testing period:")
print(test_df["order date (DateOrders)"].min(), "to",
      test_df["order date (DateOrders)"].max())

# Evaluate each saved model
models = {
    "Random Forest": "supply_chain_risk_model_v2.pkl",
    "Gradient Boosting": "supply_chain_hgb_v1.pkl",
}

for name, path in models.items():
    print(f"\n{'=' * 45}")
    print(name)

    saved = joblib.load(path)
    model = saved["model"] if isinstance(saved, dict) else saved

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
    print("F1 Score:", round(f1_score(y_test, predictions, zero_division=0), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

print("\nFuture-order evaluation completed!")
