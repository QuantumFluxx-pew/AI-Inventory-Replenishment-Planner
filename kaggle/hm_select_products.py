"""
hm_select_products.py: run this in a Kaggle Notebook attached to the
"H&M Personalized Fashion Recommendations" competition data.

It finds the H&M products with the steadiest daily sales (the ones a 7-day
moving average forecasts best) and saves ONLY what our app needs:
    /kaggle/working/hm_products.csv      names, type, colour, stats
    /kaggle/working/hm_daily_sales.csv   units sold per day, last 365 days
    /kaggle/working/hm_images/<id>.jpg   one product photo each
    /kaggle/working/hm_selection.zip     all of the above in one file

Same steadiness measures as used on M5 and Online Retail II:
    pct_days_sold - share of days with at least one sale
    scaled_MAE    - MAE of a one-day-ahead 7-day SMA / mean daily demand
"""

import glob
import os
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd

# Find the competition folder wherever Kaggle mounted it.
_found = glob.glob("/kaggle/input/**/articles.csv", recursive=True)
if not _found:
    sys.exit("Can't find articles.csv. In the right panel, click 'Add Input' and add "
             "the H&M Personalized Fashion Recommendations competition data.")
DATA = os.path.dirname(_found[0])
OUT = os.environ.get("HM_OUT", "/kaggle/working")
print("Using data in", DATA)
N_CANDIDATES = 40      # we keep 40 so we can choose the final 20 later
DAYS = 365             # analyse the last year of sales

# ---------------------------------------------------------------- load
articles = pd.read_csv(f"{DATA}/articles.csv", dtype={"article_id": str})
tx = pd.read_csv(f"{DATA}/transactions_train.csv",
                 usecols=["t_dat", "article_id", "price"],
                 dtype={"article_id": str}, parse_dates=["t_dat"])
print("transactions:", len(tx), "from", tx.t_dat.min().date(), "to", tx.t_dat.max().date())

# Keep the last year; each row is one unit sold.
start = tx.t_dat.max() - pd.Timedelta(days=DAYS - 1)
tx = tx[tx.t_dat >= start]
days = pd.date_range(start, tx.t_dat.max(), freq="D")

# ---------------------------------------------------------------- pre-filter
# Only products sold on at least 90% of days can be "steady".
sold_days = tx.groupby("article_id").t_dat.nunique()
frequent = sold_days[sold_days >= 0.9 * len(days)].index
print("articles sold on >=90% of days:", len(frequent), "of", tx.article_id.nunique())

daily = (tx[tx.article_id.isin(frequent)]
         .groupby(["t_dat", "article_id"]).size()
         .unstack(fill_value=0).reindex(days, fill_value=0))

# ---------------------------------------------------------------- score
def scaled_mae(s):
    v = s.values.astype(float)
    pred = pd.Series(v).shift(1).rolling(7).mean().values
    ok = ~np.isnan(pred)
    return np.abs(v[ok] - pred[ok]).mean() / v.mean()

stats = pd.DataFrame({
    "mean_daily_units": daily.mean(),
    "pct_days_sold": (daily > 0).mean(),
    "scaled_MAE": daily.apply(scaled_mae),
}).join(tx.groupby("article_id").price.median().rename("median_price_scaled"))

stats = stats.join(articles.set_index("article_id")[
    ["product_code", "prod_name", "product_type_name", "product_group_name",
     "colour_group_name", "department_name", "detail_desc"]])

# Must have a photo.
def image_path(aid):
    return f"{DATA}/images/{aid[:3]}/{aid}.jpg"
stats = stats[[os.path.exists(image_path(a)) for a in stats.index]]

# Steadiest first; one colour per product so the list is varied.
pick = (stats.sort_values("scaled_MAE")
        .drop_duplicates("product_code")
        .head(N_CANDIDATES))

# ---------------------------------------------------------------- save
pick.reset_index().rename(columns={"index": "article_id"}).to_csv(
    f"{OUT}/hm_products.csv", index=False)
(daily[pick.index].rename_axis("date").reset_index()
 .melt(id_vars="date", var_name="article_id", value_name="units")
 .to_csv(f"{OUT}/hm_daily_sales.csv", index=False))

os.makedirs(f"{OUT}/hm_images", exist_ok=True)
for aid in pick.index:
    shutil.copy(image_path(aid), f"{OUT}/hm_images/{aid}.jpg")

# One small zip with the two CSVs and the photos.
with zipfile.ZipFile(f"{OUT}/hm_selection.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/hm_products.csv", "hm_products.csv")
    z.write(f"{OUT}/hm_daily_sales.csv", "hm_daily_sales.csv")
    for aid in pick.index:
        z.write(f"{OUT}/hm_images/{aid}.jpg", f"hm_images/{aid}.jpg")

print("\nSummary for all frequent articles:")
print(stats[["mean_daily_units", "pct_days_sold", "scaled_MAE"]].describe().round(2))
print("\nTop candidates:")
print(pick[["prod_name", "product_type_name", "colour_group_name",
            "mean_daily_units", "pct_days_sold", "scaled_MAE"]].round(2).to_string())
print(f"\nDone. Download {OUT}/hm_selection.zip from the Output panel.")
