"""
prepare_data.py
===============

Rebuilds inventory_replenishment_results.csv from the raw M5 dataset.
This is the same pipeline as the Milestone 2 Colab notebook, as a script
so the forecasting step is part of the repository.

HOW TO RUN (on your own computer, not on Streamlit Cloud):
    1. Download sales_train_validation.csv from the Kaggle M5 Forecasting
       Accuracy competition. It is ~120 MB, too big for GitHub, so it is
       NOT committed to this repository.
    2. Put it in the same folder as this script.
    3. Run:   python prepare_data.py
    4. The script writes inventory_replenishment_results.csv, which app.py reads.

The Streamlit app does NOT need this script to run; it only reads the CSV.
"""

import argparse

import pandas as pd

import inventory_model as model

# --- Choices made in Milestone 2 -------------------------------------------
STORE_ID = "TX_1"        # one Texas Walmart store
NUM_PRODUCTS = 20        # first 20 products in that store
SMA_WINDOW = 7           # 7-day simple moving average

# Simulated stock levels: M5 has no inventory data, so these are clearly
# labelled sample values (identical to the Milestone 2 notebook).
SAMPLE_INVENTORY = [3, 10, 5, 2, 8, 15, 1, 20, 3, 10,
                    2, 8, 1, 5, 10, 25, 20, 1, 40, 5]


def build_results(sales_path: str) -> pd.DataFrame:
    sales = pd.read_csv(sales_path)
    store = sales[sales["store_id"] == STORE_ID].head(NUM_PRODUCTS)
    day_columns = [c for c in store.columns if c.startswith("d_")]

    rows = []
    for _, product in store.iterrows():
        daily_sales = product[day_columns].astype(float)
        rows.append({
            "Product_ID": product["item_id"],
            "SMA_7": model.sma_forecast(daily_sales, SMA_WINDOW),
            "MAE": model.sma_mae(daily_sales, SMA_WINDOW),
        })

    base = pd.DataFrame(rows)
    base["Current_Stock"] = SAMPLE_INVENTORY[: len(base)]

    # Run EOQ / ROP / reorder decision with the default assumptions
    # (S = $50, H = $2, lead time = 7 days, no safety stock).
    plan = model.calculate_plan(base, model.DEFAULT_ASSUMPTIONS)

    return plan[["Product_ID", "SMA_7", "MAE", "Annual_Demand", "EOQ", "ROP",
                 "Current_Stock", "Status", "Order_Quantity"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--sales", default="sales_train_validation.csv")
    parser.add_argument("--out", default="inventory_replenishment_results.csv")
    args = parser.parse_args()

    results = build_results(args.sales)
    results.to_csv(args.out, index=False)
    print(f"Saved {len(results)} products to {args.out}")
    print(model.summarize(model.calculate_plan(results)))
