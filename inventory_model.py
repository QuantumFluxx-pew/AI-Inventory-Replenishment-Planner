"""
inventory_model.py
==================

The embedded analytical models for the AI Inventory & Replenishment Planner.

Every number shown in the dashboard or said by the chatbot is calculated
HERE, in plain Python. The language model never does arithmetic; it only
decides which of these functions to call and explains the results.

Data: 20 steady-selling H&M products, one year of daily unit sales
(2019-09-24 to 2020-09-22), from the H&M Personalized Fashion
Recommendations dataset on Kaggle. See prepare_data.py.

Models:

1. Forecasting (two models)
   a. Simple Moving Average (SMA)
       d = average of the last N days of sales        (units/day, N = 7 by default)
   b. Simple Exponential Smoothing (SES)
       level_t = alpha * sales_t + (1 - alpha) * level_(t-1); forecast = latest level
       alpha is fitted per product on the first 2/3 of the year
   "Best" picks, per product, the model with the lower error on the last 1/3
   of the year (a hold-out period neither model was fitted on).
2. Forecast accuracy (MAE)
       back-test over the year: forecast each day from the days before it,
       then average the absolute errors
3. Economic Order Quantity (EOQ)
       EOQ = ceil( sqrt( 2 * D * S / H ) )
       D = annual demand = d * 365, S = cost per order, H = holding cost/unit/year
4. Reorder Point (ROP)
       ROP = ceil( d * L + safety stock )
       L = supplier lead time in days
       safety stock = z * sigma * sqrt(L)   (optional, 0 by default)
       sigma is estimated as 1.25 * MAE (standard normal-error approximation)
5. Replenishment rule
       Current_Stock <= ROP  ->  "Reorder", order EOQ units
       otherwise             ->  "Sufficient Stock", order 0
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict, replace
from statistics import NormalDist

import numpy as np
import pandas as pd

MODEL_VERSION = "hm-2026-11-24"   # must match agent.MODEL_VERSION (see app.py)

HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_CSV = os.path.join(HERE, "data", "products.csv")
SALES_CSV = os.path.join(HERE, "data", "daily_sales.csv")
IMAGES_DIR = os.path.join(HERE, "images")


# ----------------------------------------------------------------------
# 1. ASSUMPTIONS (inputs the dataset does not provide, plus model settings)
# ----------------------------------------------------------------------

# Safety-stock service levels the user may pick; 0 = no safety stock.
ALLOWED_SERVICE_LEVELS = (0, 90, 95, 97.5, 99)
# Moving-average windows the user may pick (days).
ALLOWED_SMA_WINDOWS = (3, 7, 14, 28)
# Forecast models the user may pick.
ALLOWED_FORECAST_MODELS = ("sma", "ses", "best")
FORECAST_MODEL_NAMES = {"sma": "Moving average", "ses": "Exponential smoothing",
                        "best": "Best per product"}
# Exponential smoothing: alpha values tried, and the back-test periods.
SES_ALPHAS = tuple(round(a, 2) for a in np.arange(0.05, 0.951, 0.05))
BACKTEST_START = 28          # first day scored (gives every model a warm-up)
HOLDOUT_SHARE = 1 / 3        # last third of the year is the hold-out period


class InputError(ValueError):
    """Raised when a user-supplied input is missing or invalid.

    The message is written for the end user, so the app and the chatbot
    can show it directly and ask the user to correct the value.
    """


@dataclass(frozen=True)
class Assumptions:
    """All model inputs that are assumptions or user settings."""

    ordering_cost: float = 50.0       # S: dollars per order placed
    holding_cost: float = 2.0         # H: dollars per unit per year
    lead_time_days: float = 7.0       # L: supplier lead time in days
    demand_change_pct: float = 0.0    # scenario: +20 means demand is 20% higher
    service_level: float = 0.0        # safety stock service level (%), 0 = none
    sma_window: int = 7               # days in the moving-average forecast
    forecast_model: str = "sma"       # "sma", "ses" or "best" (per product)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_ASSUMPTIONS = Assumptions()


def validate_assumptions(**values) -> Assumptions:
    """Check user inputs and return a clean Assumptions object.

    Any input not supplied keeps its current value. All problems are
    collected and reported together so the user can fix them in one go.

    Raises:
        InputError: if any value is not a number, negative, or out of range.
    """
    base = values.pop("base", DEFAULT_ASSUMPTIONS)
    problems = []
    clean = {}

    rules = {
        # name: (minimum, maximum, include_minimum, description)
        "ordering_cost": (0, 100_000, False, "Ordering cost ($ per order)"),
        "holding_cost": (0, 100_000, False, "Holding cost ($ per unit per year)"),
        "lead_time_days": (0, 365, False, "Lead time (days)"),
        "demand_change_pct": (-100, 1_000, True, "Demand change (%)"),
    }
    choices = {
        "service_level": (ALLOWED_SERVICE_LEVELS, "Service level", "%",
                          " Use 0 for no safety stock."),
        "sma_window": (ALLOWED_SMA_WINDOWS, "Moving-average window", " days", ""),
    }

    for name, value in values.items():
        if value is None:
            continue  # not supplied -> keep the current value

        if name == "forecast_model":
            key = str(value).strip().lower()
            aliases = {"moving average": "sma", "simple moving average": "sma",
                       "exponential smoothing": "ses", "simple exponential smoothing": "ses",
                       "best per product": "best", "auto": "best"}
            key = aliases.get(key, key)
            if key not in ALLOWED_FORECAST_MODELS:
                problems.append("Forecast model must be 'sma' (moving average), 'ses' "
                                f"(exponential smoothing) or 'best' (got '{value}').")
                continue
            clean[name] = key
            continue

        if name in choices:
            allowed, label, unit, hint = choices[name]
            try:
                number = float(value)
            except (TypeError, ValueError):
                problems.append(f"{label} must be a number, got '{value}'.")
                continue
            if number not in allowed:
                problems.append(
                    f"{label} must be one of "
                    + ", ".join(f"{v:g}{unit}" for v in allowed)
                    + f" (got {number:g}).{hint}"
                )
                continue
            clean[name] = int(number) if name == "sma_window" else number
            continue

        if name not in rules:
            problems.append(f"Unknown input '{name}'.")
            continue

        low, high, include_low, label = rules[name]
        try:
            number = float(value)
        except (TypeError, ValueError):
            problems.append(f"{label} must be a number, got '{value}'.")
            continue

        if math.isnan(number) or math.isinf(number):
            problems.append(f"{label} must be a finite number.")
            continue

        too_low = number < low if include_low else number <= low
        if too_low:
            sign = "at least" if include_low else "greater than"
            problems.append(f"{label} must be {sign} {low:g}, got {number:g}.")
            continue
        if number > high:
            problems.append(f"{label} must be at most {high:,g}, got {number:g}.")
            continue

        clean[name] = number

    if problems:
        raise InputError(" ".join(problems))

    return replace(base, **clean)


# ----------------------------------------------------------------------
# 2. FORECASTING MODELS
# ----------------------------------------------------------------------

def sma_forecast(daily_sales, window: int = 7) -> float:
    """Average of the last `window` days of sales (units per day)."""
    sales = pd.Series(daily_sales, dtype=float)
    if len(sales) < window:
        raise InputError(f"Need at least {window} days of sales, got {len(sales)}.")
    return float(sales.iloc[-window:].mean())


def sma_backtest(daily_sales, window: int = 7) -> pd.Series:
    """One-day-ahead SMA forecast for every day (NaN for the first `window` days).

    The forecast for day t is the mean of days t-window .. t-1.
    """
    sales = pd.Series(daily_sales, dtype=float)
    return sales.shift(1).rolling(window).mean()


def sma_mae(daily_sales, window: int = 7) -> float:
    """Mean Absolute Error of the one-day-ahead SMA forecast."""
    sales = pd.Series(daily_sales, dtype=float).reset_index(drop=True)
    predictions = sma_backtest(sales, window)
    return float((sales - predictions).abs().mean())  # NaNs are skipped


def ses_backtest(daily_sales, alpha: float, init_days: int = 7) -> np.ndarray:
    """One-day-ahead exponential-smoothing forecasts for every day.

    The level starts at the average of the first `init_days` days. The
    forecast for day t is the level after day t-1; then the level moves a
    share `alpha` of the way towards that day's actual sales.
    """
    sales = np.asarray(daily_sales, dtype=float)
    if not 0 < alpha <= 1:
        raise InputError("Smoothing constant alpha must be between 0 and 1.")
    if len(sales) < init_days:
        raise InputError(f"Need at least {init_days} days of sales, got {len(sales)}.")
    forecasts = np.empty(len(sales))
    level = sales[:init_days].mean()
    for t, actual in enumerate(sales):
        forecasts[t] = level
        level = alpha * actual + (1 - alpha) * level
    return forecasts


def ses_forecast(daily_sales, alpha: float) -> float:
    """Forecast for tomorrow: the smoothed level after the last day."""
    sales = np.asarray(daily_sales, dtype=float)
    last_forecast = ses_backtest(sales, alpha)[-1]
    return float(alpha * sales[-1] + (1 - alpha) * last_forecast)


def _mae(actual, predicted, start, end=None) -> float:
    a = np.asarray(actual, dtype=float)[start:end]
    p = np.asarray(predicted, dtype=float)[start:end]
    ok = ~np.isnan(p)
    return float(np.abs(a[ok] - p[ok]).mean())


def fit_ses_alpha(daily_sales, start: int = BACKTEST_START, end: int | None = None) -> float:
    """Alpha (from SES_ALPHAS) with the lowest back-test MAE on days start..end."""
    return min(SES_ALPHAS, key=lambda a: _mae(daily_sales, ses_backtest(daily_sales, a), start, end))


def holdout_start(n_days: int) -> int:
    """First day of the hold-out period (the last third of the history)."""
    return int(round(n_days * (1 - HOLDOUT_SHARE)))


# ----------------------------------------------------------------------
# 3. INVENTORY MODELS
# ----------------------------------------------------------------------

def _ceil(value):
    """Round up to whole units, ignoring tiny floating-point noise.

    Example: 4/7 * 7 is stored as 4.0000000001, which a plain ceil would
    turn into 5. Rounding to 9 decimals first prevents that.
    """
    return np.ceil(np.round(value, 9))


def economic_order_quantity(annual_demand, ordering_cost, holding_cost):
    """EOQ = sqrt(2DS/H), rounded up to whole units. Works on arrays."""
    if ordering_cost <= 0 or holding_cost <= 0:
        raise InputError("Ordering cost and holding cost must be greater than 0.")
    demand = np.maximum(np.asarray(annual_demand, dtype=float), 0)
    return _ceil(np.sqrt(2 * demand * ordering_cost / holding_cost)).astype(int)


def safety_stock(mae, lead_time_days, service_level):
    """Safety stock = z * sigma * sqrt(L), with sigma = 1.25 * MAE.

    Returns 0 when service_level is 0 (no safety stock).
    """
    if not service_level:
        return np.zeros_like(np.asarray(mae, dtype=float))
    z = NormalDist().inv_cdf(service_level / 100)
    sigma = 1.25 * np.asarray(mae, dtype=float)
    return z * sigma * math.sqrt(lead_time_days)


def reorder_point(daily_demand, lead_time_days, safety_units=0):
    """ROP = demand during lead time + safety stock, rounded up."""
    lead_demand = np.asarray(daily_demand, dtype=float) * lead_time_days
    return _ceil(lead_demand + safety_units).astype(int)


# ----------------------------------------------------------------------
# 4. DATA LOADING
# ----------------------------------------------------------------------

PRODUCT_COLUMNS = ["Product_ID", "Name", "Product_Type", "Colour",
                   "Department", "Description", "Current_Stock"]


def load_sales(path: str = SALES_CSV) -> pd.DataFrame:
    """Daily unit sales as a wide table: one row per day, one column per product."""
    sales = pd.read_csv(path, dtype={"article_id": str}, parse_dates=["date"])
    missing = {"date", "article_id", "units"} - set(sales.columns)
    if missing:
        raise InputError("daily_sales.csv is missing columns: " + ", ".join(sorted(missing)))
    sales["units"] = pd.to_numeric(sales["units"], errors="coerce")
    if sales["units"].isna().any() or (sales["units"] < 0).any():
        raise InputError("daily_sales.csv has missing or negative unit values.")
    wide = sales.pivot_table(index="date", columns="article_id", values="units", aggfunc="sum")
    return wide.asfreq("D").fillna(0)


def load_base_data(products_path: str = PRODUCTS_CSV,
                   sales_path: str = SALES_CSV) -> pd.DataFrame:
    """Product details + simulated stock + forecasts for every allowed window.

    For each product and each window N in ALLOWED_SMA_WINDOWS this adds
    SMA_<N> (forecast) and MAE_<N> (back-tested accuracy), so changing the
    window in the app is instant.
    """
    products = pd.read_csv(products_path, dtype={"Product_ID": str})
    missing = [c for c in PRODUCT_COLUMNS if c not in products.columns]
    if missing:
        raise InputError("products.csv is missing columns: " + ", ".join(missing))
    products = products[PRODUCT_COLUMNS].copy()
    products["Description"] = products["Description"].fillna("")
    products["Current_Stock"] = pd.to_numeric(products["Current_Stock"], errors="coerce")
    if products["Current_Stock"].isna().any() or (products["Current_Stock"] < 0).any():
        raise InputError("Current_Stock must be a non-negative number for every product.")
    if products["Product_ID"].duplicated().any():
        raise InputError("Each Product_ID must appear only once in products.csv.")

    return add_forecasts(products, load_sales(sales_path))


def add_forecasts(products: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Add both forecast models (and their accuracy) to the product table.

    `sales` is the wide daily table (one row per day, one column per
    Product_ID). Used for the H&M sample data and for uploaded data.
    """
    products = products.copy()
    no_sales = [p for p in products["Product_ID"] if p not in sales.columns]
    if no_sales:
        raise InputError("No sales history for: " + ", ".join(no_sales))

    for window in ALLOWED_SMA_WINDOWS:
        products[f"SMA_{window}"] = [sma_forecast(sales[p], window) for p in products["Product_ID"]]
        products[f"MAE_{window}"] = [sma_mae(sales[p], window) for p in products["Product_ID"]]

    # Exponential smoothing: alpha fitted on the first 2/3 of the year, then
    # both models are scored on the last 1/3 (the hold-out), which decides
    # the "best per product" choice.
    split = holdout_start(len(sales))
    for col in ["SES", "SES_ALPHA", "SES_MAE", "HOLD_SES"] + [f"HOLD_SMA_{w}" for w in ALLOWED_SMA_WINDOWS]:
        products[col] = 0.0
    for i, pid in products["Product_ID"].items():
        y = sales[pid].to_numpy(dtype=float)
        alpha = fit_ses_alpha(y, BACKTEST_START, split)
        ses_path = ses_backtest(y, alpha)
        products.at[i, "SES_ALPHA"] = alpha
        products.at[i, "SES"] = ses_forecast(y, alpha)
        products.at[i, "SES_MAE"] = _mae(y, ses_path, BACKTEST_START)
        products.at[i, "HOLD_SES"] = _mae(y, ses_path, split)
        for w in ALLOWED_SMA_WINDOWS:
            products.at[i, f"HOLD_SMA_{w}"] = _mae(y, sma_backtest(y, w).to_numpy(), split)
    return products


# ----------------------------------------------------------------------
# 4b. USER-UPLOADED DATA
# ----------------------------------------------------------------------

MIN_UPLOAD_DAYS = 60          # enough for a 28-day window, warm-up and hold-out
MAX_UPLOAD_PRODUCTS = 200
ESTIMATED_STOCK_DAYS = 10     # if no stock file: stock = 10 days of recent sales


def _find_column(frame: pd.DataFrame, options, what: str, file: str) -> str:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for option in options:
        if option in lookup:
            return lookup[option]
    raise InputError(f"The {file} needs a '{options[0]}' column ({what}). "
                     f"Columns found: {', '.join(map(str, frame.columns))}.")


def prepare_uploaded_data(sales_raw: pd.DataFrame, stock_raw: pd.DataFrame | None = None):
    """Check and convert a user's own data into the model's format.

    sales_raw: one row per product per day, with columns date, product, units
               (other names accepted: day; item / product_id / sku / name;
               quantity / qty / sales).
    stock_raw: optional, columns product, current_stock (or stock / on_hand).
               Without it, stock is estimated as 10 days of recent sales.

    Returns (products, sales_wide, notes). Raises InputError with a plain
    message for anything that would make the results wrong.
    """
    file = "sales file"
    date_col = _find_column(sales_raw, ["date", "day"], "the day of the sale", file)
    prod_col = _find_column(sales_raw, ["product", "item", "product_id", "sku", "name"],
                            "the product name or ID", file)
    unit_col = _find_column(sales_raw, ["units", "quantity", "qty", "sales"],
                            "units sold that day", file)

    sales = pd.DataFrame({
        "date": pd.to_datetime(sales_raw[date_col], errors="coerce"),
        "product": sales_raw[prod_col].astype(str).str.strip(),
        "units": pd.to_numeric(sales_raw[unit_col], errors="coerce"),
    })
    problems = []
    if sales["date"].isna().any():
        problems.append(f"{int(sales['date'].isna().sum())} rows have a date that could not be read "
                        "(use YYYY-MM-DD).")
    if sales["units"].isna().any():
        problems.append(f"{int(sales['units'].isna().sum())} rows have units that are not a number.")
    if (sales["units"] < 0).any():
        problems.append(f"{int((sales['units'] < 0).sum())} rows have negative units.")
    if (sales["product"] == "").any() or sales["product"].isin(["nan", "None"]).any():
        problems.append("Some rows have no product name.")
    if problems:
        raise InputError("Please fix the sales file: " + " ".join(problems))

    wide = (sales.pivot_table(index="date", columns="product", values="units", aggfunc="sum")
            .asfreq("D").fillna(0))
    notes = []
    if len(wide) < MIN_UPLOAD_DAYS:
        raise InputError(f"The sales file covers {len(wide)} days; at least {MIN_UPLOAD_DAYS} "
                         "days are needed to forecast and test accuracy.")
    if wide.shape[1] > MAX_UPLOAD_PRODUCTS:
        raise InputError(f"The sales file has {wide.shape[1]} products; the limit is "
                         f"{MAX_UPLOAD_PRODUCTS}.")
    missing_days = int(sales.groupby("date").size().reindex(wide.index).isna().sum())
    if missing_days:
        notes.append(f"{missing_days} days had no rows at all and were counted as zero sales.")
    never = [p for p in wide.columns if wide[p].tail(28).sum() == 0]
    if never:
        notes.append("No sales in the last 28 days, so no reorder is suggested for: "
                     + ", ".join(never[:5]) + ("…" if len(never) > 5 else "") + ".")

    products = pd.DataFrame({"Product_ID": wide.columns, "Name": wide.columns})
    products["Product_Type"] = "Uploaded"
    products["Colour"] = ""
    products["Department"] = ""
    products["Description"] = ""

    if stock_raw is not None:
        file = "stock file"
        sp = _find_column(stock_raw, ["product", "item", "product_id", "sku", "name"],
                          "the product name or ID", file)
        sq = _find_column(stock_raw, ["current_stock", "stock", "on_hand", "inventory"],
                          "units in stock now", file)
        stock = pd.Series(pd.to_numeric(stock_raw[sq], errors="coerce").values,
                          index=stock_raw[sp].astype(str).str.strip())
        if stock.isna().any() or (stock < 0).any():
            raise InputError("Please fix the stock file: every stock value must be a number, 0 or more.")
        missing = [p for p in wide.columns if p not in stock.index]
        if missing:
            raise InputError("The stock file has no stock for: " + ", ".join(missing[:5])
                             + ("…" if len(missing) > 5 else "") + ".")
        extra = [p for p in stock.index if p not in wide.columns]
        if extra:
            notes.append("Ignored stock for products with no sales: " + ", ".join(extra[:5]) + ".")
        products["Current_Stock"] = products["Product_ID"].map(stock).astype(float)
    else:
        recent = wide.tail(7).mean()
        products["Current_Stock"] = np.round(products["Product_ID"].map(recent) * ESTIMATED_STOCK_DAYS)
        notes.append(f"No stock file: stock was estimated as {ESTIMATED_STOCK_DAYS} days of recent "
                     "sales for every product. Upload a stock file for real recommendations.")

    return add_forecasts(products, wide), wide, notes


# ----------------------------------------------------------------------
# 5. FULL REPLENISHMENT PLAN
# ----------------------------------------------------------------------

def calculate_plan(base: pd.DataFrame, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> pd.DataFrame:
    """Run SMA -> EOQ -> ROP -> reorder decision for every product.

    Returns one row per product with inputs, results and helper columns
    (Days_of_Cover, Shortfall) that explain each decision.
    """
    a = assumptions
    plan = base.copy()

    # Forecast from the chosen model, adjusted for any what-if demand change.
    # (The column keeps the name "SMA" for the forecast; Forecast_Model says
    # which model produced it.)
    use_ses = np.full(len(plan), a.forecast_model == "ses")
    if a.forecast_model == "best":
        use_ses = (plan["HOLD_SES"] < plan[f"HOLD_SMA_{a.sma_window}"]).to_numpy()
    sma_label = f"{a.sma_window}-day moving average"
    plan["SMA"] = np.where(use_ses, plan["SES"], plan[f"SMA_{a.sma_window}"])
    plan["MAE"] = np.where(use_ses, plan["SES_MAE"], plan[f"MAE_{a.sma_window}"])
    plan["Forecast_Model"] = [
        f"Exponential smoothing (alpha {al:g})" if s else sma_label
        for s, al in zip(use_ses, plan["SES_ALPHA"])]
    plan["Holdout_MAE"] = np.where(use_ses, plan["HOLD_SES"], plan[f"HOLD_SMA_{a.sma_window}"])
    plan["Daily_Demand"] = plan["SMA"] * (1 + a.demand_change_pct / 100)
    plan["Annual_Demand"] = plan["Daily_Demand"] * 365

    # Inventory models.
    plan["EOQ"] = economic_order_quantity(plan["Annual_Demand"], a.ordering_cost, a.holding_cost)
    plan["Safety_Stock"] = np.round(
        safety_stock(plan["MAE"], a.lead_time_days, a.service_level), 2)
    plan["ROP"] = reorder_point(plan["Daily_Demand"], a.lead_time_days, plan["Safety_Stock"])

    # Replenishment decision: reorder when stock <= ROP.
    needs_order = (plan["Current_Stock"] <= plan["ROP"]) & (plan["Daily_Demand"] > 0)
    plan["Status"] = np.where(needs_order, "Reorder", "Sufficient Stock")
    plan["Order_Quantity"] = np.where(needs_order, plan["EOQ"], 0).astype(int)

    # Explanations: how many days the stock lasts, and how far below ROP it is.
    with np.errstate(divide="ignore"):
        cover = plan["Current_Stock"] / plan["Daily_Demand"]
    plan["Days_of_Cover"] = cover.replace(np.inf, np.nan).round(1)
    plan["Shortfall"] = (plan["ROP"] - plan["Current_Stock"]).clip(lower=0).astype(int)
    # Sales at risk: demand during the lead time that current stock cannot cover,
    # i.e. what would be lost if an order placed today arrives after L days.
    plan["Units_At_Risk"] = _ceil(
        (plan["Daily_Demand"] * a.lead_time_days - plan["Current_Stock"]).clip(lower=0)
    ).astype(int)

    # Annual cost of the EOQ policy: ordering + holding cost.
    with np.errstate(divide="ignore", invalid="ignore"):
        orders_per_year = np.where(plan["EOQ"] > 0, plan["Annual_Demand"] / plan["EOQ"], 0)
    plan["Annual_Cost"] = np.round(
        orders_per_year * a.ordering_cost + plan["EOQ"] / 2 * a.holding_cost, 2)

    return plan


def cost_curve(annual_demand, ordering_cost, holding_cost, points=60):
    """Ordering, holding and total yearly cost for a range of order sizes
    around the EOQ (from 20% to 300% of it). Shows why the EOQ is the
    cheapest order size: there the two costs meet and the total is lowest."""
    if ordering_cost <= 0 or holding_cost <= 0:
        raise InputError("Ordering cost and holding cost must be greater than 0.")
    if annual_demand <= 0:
        raise InputError("This product has no forecast demand, so there is no cost curve.")
    eoq = math.sqrt(2 * annual_demand * ordering_cost / holding_cost)
    q = np.linspace(0.2 * eoq, 3 * eoq, points)
    ordering = annual_demand / q * ordering_cost
    holding = q / 2 * holding_cost
    return pd.DataFrame({"Order_Size": q, "Ordering_Cost": ordering,
                         "Holding_Cost": holding, "Total_Cost": ordering + holding})


def safety_stock_tradeoff(base: pd.DataFrame, assumptions: "Assumptions") -> pd.DataFrame:
    """For each allowed service level: total safety stock (units) across all
    products, its yearly holding cost, and how many products need reordering.
    Everything else stays at the current assumptions."""
    rows = []
    for level in ALLOWED_SERVICE_LEVELS:
        plan = calculate_plan(base, replace(assumptions, service_level=level))
        units = float(plan["Safety_Stock"].sum())
        rows.append({"Service_Level": level,
                     "Safety_Stock_Units": round(units),
                     "Extra_Holding_Cost": round(units * assumptions.holding_cost, 2),
                     "Need_Reorder": int((plan["Status"] == "Reorder").sum())})
    return pd.DataFrame(rows)


def summarize(plan: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard and the chatbot."""
    reorder = plan[plan["Status"] == "Reorder"]
    return {
        "products": int(len(plan)),
        "need_reorder": int(len(reorder)),
        "sufficient_stock": int((plan["Status"] == "Sufficient Stock").sum()),
        "recommended_order_units": int(reorder["Order_Quantity"].sum()),
        "current_units_on_hand": int(plan["Current_Stock"].sum()),
        "total_annual_policy_cost": round(float(plan["Annual_Cost"].sum()), 2),
    }


def prioritize(plan: pd.DataFrame) -> pd.DataFrame:
    """Rank products that need reordering, most urgent first.

    Urgency = fewest days of stock left. Ties are broken by the
    larger shortfall below the reorder point, then by higher demand.
    """
    reorder = plan[plan["Status"] == "Reorder"].copy()
    return reorder.sort_values(
        ["Days_of_Cover", "Shortfall", "Daily_Demand"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def compare_plans(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side changes in ROP, EOQ, status and order quantity."""
    cols = ["ROP", "EOQ", "Status", "Order_Quantity"]
    merged = before[["Product_ID", "Name", "Current_Stock"] + cols].merge(
        after[["Product_ID"] + cols], on="Product_ID", suffixes=("_before", "_after"))
    merged["Status_Changed"] = merged["Status_before"] != merged["Status_after"]
    return merged


# ----------------------------------------------------------------------
# 6. PRODUCT LOOKUP (accepts loose user typing)
# ----------------------------------------------------------------------

def find_product(plan: pd.DataFrame, text: str) -> str:
    """Turn user text into a Product_ID.

    Accepts an H&M article ID (with or without the leading zero), the
    product name, or a unique part of it ('scallop socks', 'brit tee').

    Raises:
        InputError: if nothing or more than one product matches.
    """
    if not text or not str(text).strip():
        raise InputError("Please tell me which product, for example 'Scallop 5p Socks'.")
    query = str(text).strip()
    ids = list(plan["Product_ID"])

    # 1) Article ID, e.g. '0608776002' or '608776002'.
    digits = query.lstrip("0")
    if digits.isdigit():
        hits = [pid for pid in ids if pid.lstrip("0") == digits]
        if len(hits) == 1:
            return hits[0]

    # 2) Exact name, then names containing every word the user typed.
    names = dict(zip(plan["Product_ID"], plan["Name"]))
    lower = query.lower()
    exact = [pid for pid, name in names.items() if name.lower() == lower]
    if len(exact) == 1:
        return exact[0]
    words = [w for w in lower.replace("-", " ").split() if w not in {"the", "a", "of"}]
    hits = [pid for pid, name in names.items()
            if words and all(w in name.lower() for w in words)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        # 3) Product type, e.g. 'socks', 'cardigan', 't-shirt'.
        types = dict(zip(plan["Product_ID"], plan["Product_Type"].str.lower()))
        wanted = lower.rstrip("s")
        hits = [pid for pid, kind in types.items()
                if wanted in {kind.rstrip("s"), *[k.strip().rstrip("s") for k in kind.split("/")]}]
        if len(hits) == 1:
            return hits[0]
    if len(hits) > 1:
        options = ", ".join(names[p] for p in hits[:5])
        raise InputError(f"'{text}' matches several products ({options}). Which one?")

    raise InputError(
        f"I can't find a product called '{text}'. Try a name such as "
        f"'{names[ids[0]]}' or an article ID such as {ids[0]}."
    )


def image_path(product_id: str) -> str:
    """Local path of the product's photo thumbnail."""
    return os.path.join(IMAGES_DIR, f"{product_id}.jpg")
