
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the existing trained model
model = joblib.load("supply_chain_risk_model_v2.pkl")

# These must match the features used during training
NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "Type",
    "Category Name",
    "Customer Segment",
    "Market",
    "Order Region",
    "Shipping Mode",
    "Department Name"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@app.route("/api/predict-risk", methods=["POST"])
def predict_risk():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object"}), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing required features",
            "missingFeatures": missing
        }), 400

    try:
        input_data = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        risk_probability = float(
            probabilities[classes.index(1)]
        ) if 1 in classes else 0.0

        return jsonify({
            "prediction": prediction,
            "riskLabel": (
                "Late-Delivery Risk"
                if prediction == 1
                else "No Late-Delivery Risk"
            ),
            "riskProbability": round(risk_probability * 100, 2)
        })

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model": "Random Forest",
        "task": "Late-delivery risk prediction"
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
