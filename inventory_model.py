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

1. Simple Moving Average (SMA) forecast
       d = average of the last N days of sales        (units/day, N = 7 by default)
2. Forecast accuracy (MAE)
       back-test over the whole year: for each day, forecast it with the
       average of the N days before it, then average the absolute errors
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

MODEL_VERSION = "hm-2026-11-10"   # must match agent.MODEL_VERSION (see app.py)

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

    sales = load_sales(sales_path)
    no_sales = [p for p in products["Product_ID"] if p not in sales.columns]
    if no_sales:
        raise InputError("No sales history for: " + ", ".join(no_sales))

    for window in ALLOWED_SMA_WINDOWS:
        products[f"SMA_{window}"] = [sma_forecast(sales[p], window) for p in products["Product_ID"]]
        products[f"MAE_{window}"] = [sma_mae(sales[p], window) for p in products["Product_ID"]]
    return products


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

    # Forecast for the chosen window, adjusted for any what-if demand change.
    plan["SMA"] = plan[f"SMA_{a.sma_window}"]
    plan["MAE"] = plan[f"MAE_{a.sma_window}"]
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

    # Annual cost of the EOQ policy: ordering + holding cost.
    with np.errstate(divide="ignore", invalid="ignore"):
        orders_per_year = np.where(plan["EOQ"] > 0, plan["Annual_Demand"] / plan["EOQ"], 0)
    plan["Annual_Cost"] = np.round(
        orders_per_year * a.ordering_cost + plan["EOQ"] / 2 * a.holding_cost, 2)

    return plan


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
