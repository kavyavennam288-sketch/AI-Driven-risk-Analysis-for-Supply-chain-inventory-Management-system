
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
path = r"archive (1)\DataCoSupplyChainDataset.csv"
df = pd.read_csv(path, encoding="latin1")

df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"], errors="coerce"
)
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df = df.dropna(subset=["order date (DateOrders)", "Sales"])

# Aggregate sales by day
daily = (
    df.groupby(df["order date (DateOrders)"].dt.floor("D"))["Sales"]
    .sum()
    .sort_index()
    .rename_axis("date")
    .reset_index()
)

# Fill missing calendar dates with zero sales
calendar = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
daily = (
    daily.set_index("date")
    .reindex(calendar, fill_value=0)
    .rename_axis("date")
    .reset_index()
)

# Features based only on past sales
daily["day_of_week"] = daily["date"].dt.day_of_week
daily["month"] = daily["date"].dt.month
daily["day_of_year"] = daily["date"].dt.day_of_year
daily["lag_1"] = daily["Sales"].shift(1)
daily["lag_7"] = daily["Sales"].shift(7)
daily["avg_7"] = daily["Sales"].shift(1).rolling(7).mean()
daily["avg_30"] = daily["Sales"].shift(1).rolling(30).mean()

daily = daily.dropna().reset_index(drop=True)

features = [
    "day_of_week", "month", "day_of_year",
    "lag_1", "lag_7", "avg_7", "avg_30"
]

# Four consecutive 28-day holdouts
start = pd.Timestamp("2017-04-01")
window_days = 28
all_actual = []
all_gb_predictions = []
all_naive_predictions = []

for fold in range(4):
    test_start = start + pd.Timedelta(days=fold * window_days)
    test_end = test_start + pd.Timedelta(days=window_days)

    train = daily[daily["date"] < test_start]
    test = daily[
        (daily["date"] >= test_start) &
        (daily["date"] < test_end)
    ]

    if len(test) != window_days or len(train) == 0:
        print("Insufficient data for fold", fold + 1)
        continue

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(train[features], train["Sales"])
    predictions = model.predict(test[features])

    # Previous-week forecast; evaluated one day at a time
    naive = test["lag_7"].to_numpy()

    actual = test["Sales"].to_numpy()
    wape_gb = np.abs(actual - predictions).sum() / np.abs(actual).sum() * 100
    wape_naive = np.abs(actual - naive).sum() / np.abs(actual).sum() * 100

    print(f"\nFold {fold + 1}: {test_start.date()} to "
          f"{(test_end - pd.Timedelta(days=1)).date()}")
    print(f"Gradient Boosting WAPE: {wape_gb:.2f}%")
    print(f"Previous-week WAPE: {wape_naive:.2f}%")

    all_actual.extend(actual)
    all_gb_predictions.extend(predictions)
    all_naive_predictions.extend(naive)

if all_actual:
    actual = np.array(all_actual)
    gb = np.array(all_gb_predictions)
    naive = np.array(all_naive_predictions)

    print("\n========== OVERALL RESULTS ==========")
    print("Gradient Boosting WAPE:",
          round(np.abs(actual - gb).sum() / np.abs(actual).sum() * 100, 2), "%")
    print("Previous-week WAPE:",
          round(np.abs(actual - naive).sum() / np.abs(actual).sum() * 100, 2), "%")
    print("Gradient Boosting MAE:", round(mean_absolute_error(actual, gb), 2))
    print("Evaluated days:", len(actual))
