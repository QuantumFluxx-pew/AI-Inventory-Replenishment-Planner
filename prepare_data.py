"""
prepare_data.py
===============

Builds the app's data files from the H&M Personalized Fashion
Recommendations dataset (Kaggle).

Two steps:

STEP 1 (on Kaggle): run kaggle/hm_select_products.py in a Kaggle Notebook
    attached to the competition data. It reads all 31 million purchases,
    keeps the last 365 days, finds the 40 steadiest-selling products and
    saves hm_selection.zip (names, daily sales, photos). Only ~7 MB.

STEP 2 (here): unzip hm_selection.zip into a folder and run
        python prepare_data.py --input path/to/unzipped/folder
    This script
      * checks the sales data for gaps, negatives and duplicates,
      * skips products being phased out (last 4 weeks < 50% of yearly average),
      * picks up to 20 products (steadiest first, at most 3 per type),
      * shrinks each photo to a 300-pixel thumbnail,
      * adds a SIMULATED current stock level (H&M has no inventory data),
      * writes data/products.csv, data/daily_sales.csv and images/<id>.jpg.

The app then calculates the SMA forecast, MAE, EOQ and reorder point
live from data/daily_sales.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

N_PRODUCTS = 20
MAX_PER_TYPE = 3          # keep the product mix varied
THUMB_WIDTH = 300         # pixels
STOCK_SEED = 42           # fixed seed -> the same simulated stock every run
MIN_RECENT_RATIO = 0.5    # last 4 weeks must be >= 50% of the yearly average
LEAD_TIME_DAYS = 7        # used only to make a realistic stock mix


def still_active(sales: pd.DataFrame, min_ratio: float = MIN_RECENT_RATIO) -> pd.Series:
    """True for products whose last-28-day average sales are at least
    `min_ratio` of their yearly average. Products below that are being
    phased out, so planning replenishment for them makes no sense."""
    wide = sales.pivot_table(index="date", columns="article_id", values="units", aggfunc="sum")
    return wide.tail(28).mean() >= min_ratio * wide.mean()


def select_products(products: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Steadiest first (lowest scaled MAE), at most MAX_PER_TYPE per type.

    Skips products whose type H&M lists as 'Unknown' and products that
    are being phased out (see still_active)."""
    active = still_active(sales)
    dropped = products[~products["article_id"].map(active).fillna(False)]
    if len(dropped):
        print("Skipped (being phased out):", ", ".join(dropped["prod_name"].str.strip()))
    ranked = products[(products["product_type_name"] != "Unknown")
                      & products["article_id"].map(active).fillna(False)].sort_values("scaled_MAE")
    chosen = ranked.groupby("product_type_name", group_keys=False).head(MAX_PER_TYPE)
    return chosen.sort_values("scaled_MAE").head(N_PRODUCTS)


def clean_sales(sales: pd.DataFrame, ids) -> pd.DataFrame:
    """Keep the chosen products and check the series are complete."""
    sales = sales[sales["article_id"].isin(ids)].copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales["units"] = pd.to_numeric(sales["units"], errors="coerce")

    if sales["units"].isna().any():
        raise ValueError("Non-numeric unit values found in daily sales.")
    if (sales["units"] < 0).any():
        raise ValueError("Negative unit values found in daily sales.")
    if sales.duplicated(["date", "article_id"]).any():
        raise ValueError("Duplicate (date, product) rows found in daily sales.")

    # Every product must have one row for every day (zero = no sales).
    days = pd.date_range(sales["date"].min(), sales["date"].max(), freq="D")
    full = (sales.pivot(index="date", columns="article_id", values="units")
            .reindex(days).fillna(0).astype(int))
    full.index.name = "date"
    return full.reset_index().melt(id_vars="date", var_name="article_id", value_name="units")


def simulate_stock(daily_demand: pd.Series) -> pd.Series:
    """SIMULATED stock: each product gets between 3 and 12 days of cover
    at its recent demand. With a 7-day lead time, a bit under half of
    products end up at or below their reorder point, a realistic mix."""
    rng = np.random.default_rng(STOCK_SEED)
    days_cover = rng.uniform(3, 12, size=len(daily_demand))
    return np.round(daily_demand.values * days_cover).astype(int)


def make_thumbnail(src: str, dst: str) -> None:
    from PIL import Image
    with Image.open(src) as img:
        img = img.convert("RGB")
        height = round(img.height * THUMB_WIDTH / img.width)
        img.resize((THUMB_WIDTH, height), Image.LANCZOS).save(dst, "JPEG", quality=85)


def main(input_dir: str, out_dir: str) -> None:
    products = pd.read_csv(os.path.join(input_dir, "hm_products.csv"), dtype={"article_id": str})
    sales = pd.read_csv(os.path.join(input_dir, "hm_daily_sales.csv"), dtype={"article_id": str})

    sales = clean_sales(sales, products["article_id"])
    chosen = select_products(products, sales)
    sales = sales[sales["article_id"].isin(chosen["article_id"])]

    # Recent demand (last 7 days) is only used to size the simulated stock.
    last7 = (sales[sales["date"] > sales["date"].max() - pd.Timedelta(days=7)]
             .groupby("article_id")["units"].mean())

    table = pd.DataFrame({
        "Product_ID": chosen["article_id"].values,
        "Name": chosen["prod_name"].str.strip().str.rstrip(".").values,
        "Product_Type": chosen["product_type_name"].values,
        "Colour": chosen["colour_group_name"].values,
        "Department": chosen["department_name"].values,
        "Description": chosen["detail_desc"].fillna("").values,
    })
    table["Current_Stock"] = simulate_stock(last7.reindex(table["Product_ID"]))

    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    table.to_csv(os.path.join(out_dir, "data", "products.csv"), index=False)
    sales.to_csv(os.path.join(out_dir, "data", "daily_sales.csv"), index=False,
                 date_format="%Y-%m-%d")

    for pid in table["Product_ID"]:
        make_thumbnail(os.path.join(input_dir, "hm_images", f"{pid}.jpg"),
                       os.path.join(out_dir, "images", f"{pid}.jpg"))

    print(f"Saved {len(table)} products, {len(sales)} daily sales rows "
          f"({sales['date'].min().date()} to {sales['date'].max().date()}), "
          f"{len(table)} thumbnails.")
    print(table[["Product_ID", "Name", "Product_Type", "Current_Stock"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build app data from the H&M selection.")
    parser.add_argument("--input", required=True, help="Folder with the unzipped hm_selection.zip")
    parser.add_argument("--out", default=".", help="Project folder (default: current folder)")
    args = parser.parse_args()
    main(args.input, args.out)
