"""
Automated tests for the AI Inventory & Replenishment Planner.

Run from the project folder:   python -m pytest -v

They check:
  1. The embedded models reproduce the Milestone 2 results exactly.
  2. The formulas are mathematically correct on hand-checked examples.
  3. Invalid or missing inputs are rejected with clear messages.
  4. The agent tools return correct data and ask for missing inputs.
"""

import math
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inventory_model as m  # noqa: E402
from agent import InventoryTools, OfflineAssistant  # noqa: E402

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "inventory_replenishment_results.csv")


@pytest.fixture
def base():
    return m.load_base_data(CSV)


@pytest.fixture
def tools(base):
    return InventoryTools(base, m.DEFAULT_ASSUMPTIONS)


# ---------------------------------------------------------------- 1. baseline
def test_reproduces_milestone2_csv(base):
    plan = m.calculate_plan(base)
    original = pd.read_csv(CSV)
    for col in ["EOQ", "ROP", "Status", "Order_Quantity"]:
        assert list(plan[col]) == list(original[col]), col
    assert all(abs(plan["Annual_Demand"] - original["Annual_Demand"]) < 1e-9)


def test_baseline_summary(base):
    s = m.summarize(m.calculate_plan(base))
    assert (s["need_reorder"], s["sufficient_stock"], s["recommended_order_units"]) == (12, 8, 1856)


# ---------------------------------------------------------------- 2. formulas
def test_eoq_hand_calculation():
    # sqrt(2 * 1000 * 50 / 2) = sqrt(50000) = 223.6 -> 224
    assert m.economic_order_quantity([1000], 50, 2)[0] == 224
    assert m.economic_order_quantity([0], 50, 2)[0] == 0


def test_rop_ignores_float_noise():
    # 4/7 * 7 is 4.0000000001 in floating point; must stay 4, not 5.
    assert m.reorder_point([4 / 7], 7)[0] == 4
    assert m.reorder_point([1.5], 3)[0] == 5   # 4.5 rounds up to 5


def test_safety_stock_formula():
    ss = m.safety_stock([1.0], 9, 95)[0]
    expected = 1.6448536 * 1.25 * 1.0 * 3   # z * 1.25 * MAE * sqrt(9)
    assert ss == pytest.approx(expected, rel=1e-6)
    assert m.safety_stock([1.0], 9, 0)[0] == 0


def test_sma_and_mae():
    sales = [0, 0, 0, 0, 0, 0, 0, 7]
    assert m.sma_forecast(sales, 7) == 1.0
    # only day 8 has a forecast: mean of days 1-7 = 0, error = 7
    assert m.sma_mae(sales, 7) == 7.0


def test_demand_increase_raises_eoq_and_rop(base):
    before = m.calculate_plan(base)
    after = m.calculate_plan(base, m.validate_assumptions(demand_change_pct=20))
    assert (after["EOQ"] >= before["EOQ"]).all()
    assert (after["ROP"] >= before["ROP"]).all()
    # EOQ scales with sqrt(1.2) for HOBBIES_1_019
    row = after[after["Product_ID"] == "HOBBIES_1_019"].iloc[0]
    assert row["EOQ"] == math.ceil(math.sqrt(2 * 7.571428571428571 * 1.2 * 365 * 50 / 2))


def test_longer_lead_time(base):
    s = m.summarize(m.calculate_plan(base, m.validate_assumptions(lead_time_days=14)))
    assert s["need_reorder"] == 14   # HOBBIES_1_008 and _017 are added


# ---------------------------------------------------------------- 3. validation
@pytest.mark.parametrize("bad", [
    {"lead_time_days": -3}, {"lead_time_days": 0}, {"ordering_cost": 0},
    {"holding_cost": "abc"}, {"demand_change_pct": -150}, {"service_level": 80},
])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(m.InputError):
        m.validate_assumptions(**bad)


def test_product_lookup(base):
    plan = m.calculate_plan(base)
    assert m.find_product(plan, "hobbies 1 16") == "HOBBIES_1_016"
    assert m.find_product(plan, "HOBBIES-1-019") == "HOBBIES_1_019"
    with pytest.raises(m.InputError):
        m.find_product(plan, "HOBBIES_1_099")


# ---------------------------------------------------------------- 4. agent tools
def test_tool_errors_are_returned_not_raised(tools):
    result = tools.call("run_what_if", {"lead_time_days": -5})
    assert "error" in result and result["needs_user_input"]


def test_what_if_does_not_change_dashboard(tools):
    tools.call("run_what_if", {"demand_change_pct": 20})
    assert tools.assumptions == m.DEFAULT_ASSUMPTIONS


def test_update_assumptions_changes_state(tools):
    tools.call("update_assumptions", {"lead_time_days": 14})
    assert tools.assumptions.lead_time_days == 14


def test_priority_order(tools):
    ranked = tools.call("prioritize_reorders", {"limit": 3})["ranked"]
    assert [r["product_id"] for r in ranked] == ["HOBBIES_1_013", "HOBBIES_1_004", "HOBBIES_1_014"]


def test_offline_asks_for_missing_percentage(tools):
    reply = OfflineAssistant().ask("What happens if demand increases?", [], tools)
    assert "percentage" in reply.text and reply.tool_calls == []
