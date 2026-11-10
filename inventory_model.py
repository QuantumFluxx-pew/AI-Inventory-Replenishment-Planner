"""
inventory_model.py
==================

The embedded analytical models for the AI Inventory & Replenishment Planner.

Every number shown in the dashboard or said by the chatbot is calculated
HERE, in plain Python. The language model never does arithmetic; it only
decides which of these functions to call and explains the results.

Models (same formulas as the Milestone 2 Colab notebook):

1. Simple Moving Average (SMA) forecast
       d = average of the last 7 days of sales          (units/day)
2. Forecast accuracy (MAE)
       one-day-ahead rolling 7-day SMA vs. actual sales
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

With the default assumptions (S = $50, H = $2, L = 7 days, no safety stock,
no demand change) this module reproduces inventory_replenishment_results.csv
exactly. tests/test_inventory_model.py checks that.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict, replace
from statistics import NormalDist

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. ASSUMPTIONS (the inputs the M5 dataset does not provide)
# ----------------------------------------------------------------------

# Service levels the user may pick for safety stock.
# 0 means "no safety stock", which matches the Milestone 2 model.
ALLOWED_SERVICE_LEVELS = (0, 90, 95, 97.5, 99)


class InputError(ValueError):
    """Raised when a user-supplied input is missing or invalid.

    The message is written for the end user, so the app and the chatbot
    can show it directly and ask the user to correct the value.
    """


@dataclass(frozen=True)
class Assumptions:
    """All model inputs that are assumptions rather than M5 data."""

    ordering_cost: float = 50.0       # S: dollars per order placed
    holding_cost: float = 2.0         # H: dollars per unit per year
    lead_time_days: float = 7.0       # L: supplier lead time in days
    demand_change_pct: float = 0.0    # scenario: +20 means demand is 20% higher
    service_level: float = 0.0        # safety stock service level (%), 0 = none

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_ASSUMPTIONS = Assumptions()


def validate_assumptions(**values) -> Assumptions:
    """Check user inputs and return a clean Assumptions object.

    Any input not supplied keeps its default value. All problems are
    collected and reported together so the user can fix them in one go.

    Raises:
        InputError: if any value is missing a number, negative, or out of range.
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

    for name, value in values.items():
        if value is None:
            continue  # not supplied -> keep the current value

        if name == "service_level":
            try:
                level = float(value)
            except (TypeError, ValueError):
                problems.append(f"Service level must be a number, got '{value}'.")
                continue
            if level not in ALLOWED_SERVICE_LEVELS:
                problems.append(
                    "Service level must be one of "
                    + ", ".join(f"{v:g}%" for v in ALLOWED_SERVICE_LEVELS)
                    + f" (got {level:g}). Use 0 for no safety stock."
                )
                continue
            clean[name] = level
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
# 2. FORECASTING MODELS (used by prepare_data.py on the raw M5 files)
# ----------------------------------------------------------------------

def sma_forecast(daily_sales, window: int = 7) -> float:
    """Average of the last `window` days of sales (units per day)."""
    sales = pd.Series(daily_sales, dtype=float)
    if len(sales) < window:
        raise InputError(f"Need at least {window} days of sales, got {len(sales)}.")
    return float(sales.iloc[-window:].mean())


def sma_mae(daily_sales, window: int = 7) -> float:
    """Mean Absolute Error of a one-day-ahead rolling SMA forecast.

    For each day t, the forecast is the mean of days t-window .. t-1.
    This is the same back-test used in the Milestone 2 notebook.
    """
    sales = pd.Series(daily_sales, dtype=float).reset_index(drop=True)
    predictions = sales.shift(1).rolling(window).mean()
    return float((sales - predictions).abs().mean())  # NaNs are skipped


# ----------------------------------------------------------------------
# 3. INVENTORY MODELS
# ----------------------------------------------------------------------

def _ceil(value):
    """Round up to whole units, ignoring tiny floating-point noise.

    Example: 0.5714285714 * 7 is stored as 4.0000000001, which a plain
    ceil would turn into 5. Rounding to 9 decimals first prevents that.
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
# 4. FULL REPLENISHMENT PLAN
# ----------------------------------------------------------------------

BASE_COLUMNS = ["Product_ID", "SMA_7", "MAE", "Current_Stock"]


def load_base_data(path: str = "inventory_replenishment_results.csv") -> pd.DataFrame:
    """Load the per-product inputs: forecast, accuracy and (simulated) stock.

    Only Product_ID, SMA_7, MAE and Current_Stock are used as inputs.
    EOQ, ROP, Status and Order_Quantity are recalculated live by
    calculate_plan(), so they always match the current assumptions.
    """
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    missing = [col for col in BASE_COLUMNS if col not in data.columns]
    if missing:
        raise InputError("The inventory CSV is missing columns: " + ", ".join(missing))

    base = data[BASE_COLUMNS].copy()
    base["Product_ID"] = base["Product_ID"].astype(str).str.strip()

    for col in ["SMA_7", "MAE", "Current_Stock"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    bad = base[base[["SMA_7", "MAE", "Current_Stock"]].isna().any(axis=1)]
    if not bad.empty:
        raise InputError(
            "Non-numeric or empty values for: " + ", ".join(bad["Product_ID"])
        )
    if (base[["SMA_7", "MAE", "Current_Stock"]] < 0).any().any():
        raise InputError("SMA_7, MAE and Current_Stock cannot be negative.")
    if base["Product_ID"].duplicated().any():
        raise InputError("Each Product_ID must appear only once in the CSV.")

    # Friendly display name (M5 has no real product names).
    base["Name"] = base["Product_ID"].map(product_label)

    # Real shelf price from M5 sell_prices.csv, if prepare_data.py saved it.
    # Display only: prices do not change EOQ or ROP.
    prices_path = os.path.join(os.path.dirname(os.path.abspath(path)), "product_prices.csv")
    if os.path.exists(prices_path):
        prices = pd.read_csv(prices_path)[["Product_ID", "Sell_Price"]]
        base = base.merge(prices, on="Product_ID", how="left")
    else:
        base["Sell_Price"] = np.nan

    return base


def product_label(product_id: str) -> str:
    """'HOBBIES_1_019' -> 'Hobbies Item 019'.

    M5 anonymises products, so there are no real names. The label just
    makes the ID easier to read: category + item number.
    """
    parts = str(product_id).split("_")
    if len(parts) == 3:
        return f"{parts[0].title()} Item {parts[2]}"
    return str(product_id)


def calculate_plan(base: pd.DataFrame, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> pd.DataFrame:
    """Run SMA -> EOQ -> ROP -> reorder decision for every product.

    Returns one row per product with all inputs, results and helper columns
    (Days_of_Cover, Shortfall) that explain each decision.
    """
    a = assumptions
    plan = base.copy()

    # Demand forecast, adjusted for any what-if demand change.
    plan["Daily_Demand"] = plan["SMA_7"] * (1 + a.demand_change_pct / 100)
    plan["Annual_Demand"] = plan["Daily_Demand"] * 365

    # Inventory models.
    plan["EOQ"] = economic_order_quantity(plan["Annual_Demand"], a.ordering_cost, a.holding_cost)
    plan["Safety_Stock"] = np.round(
        safety_stock(plan["MAE"], a.lead_time_days, a.service_level), 2
    )
    plan["ROP"] = reorder_point(plan["Daily_Demand"], a.lead_time_days, plan["Safety_Stock"])

    # Replenishment decision (same rule as the notebook: stock <= ROP).
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
        orders_per_year * a.ordering_cost + plan["EOQ"] / 2 * a.holding_cost, 2
    )

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
    merged = before[["Product_ID", "Current_Stock"] + cols].merge(
        after[["Product_ID"] + cols], on="Product_ID", suffixes=("_before", "_after")
    )
    merged["Status_Changed"] = merged["Status_before"] != merged["Status_after"]
    return merged


# ----------------------------------------------------------------------
# 5. PRODUCT ID LOOKUP (accepts loose user typing)
# ----------------------------------------------------------------------

def find_product(plan: pd.DataFrame, text: str) -> str:
    """Turn user text like 'hobbies 1 16' or 'HOBBIES-1-016' into a Product_ID.

    Raises:
        InputError: if nothing or more than one product matches.
    """
    if not text or not str(text).strip():
        raise InputError("Please tell me which product, for example HOBBIES_1_001.")

    ids = list(plan["Product_ID"])
    query = str(text).strip().upper().replace("-", "_").replace(" ", "_")
    # Also accept the friendly name, e.g. 'Hobbies Item 019'.
    query = query.replace("_ITEM_", "_1_")

    if query in ids:
        return query

    # Allow 'HOBBIES_1_16' -> 'HOBBIES_1_016' and bare numbers like '16'.
    parts = query.split("_")
    if parts[-1].isdigit():
        number = int(parts[-1])
        candidates = [
            pid for pid in ids
            if int(pid.split("_")[-1]) == number
            and (len(parts) == 1 or pid.startswith("_".join(parts[:-1]) + "_"))
        ]
        if len(candidates) == 1:
            return candidates[0]

    raise InputError(
        f"I can't find a product called '{text}'. Valid IDs run from "
        f"{ids[0]} to {ids[-1]} (for example {ids[0]})."
    )
