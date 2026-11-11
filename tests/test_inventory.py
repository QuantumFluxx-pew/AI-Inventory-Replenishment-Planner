"""
Automated tests for the AI Inventory & Replenishment Planner (H&M data).

Run from the project folder:   python -m pytest -v

They check:
  1. The data files are complete and clean.
  2. The forecasting and inventory formulas are correct on hand-checked examples.
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

BRIT_TEE = "0624486001"    # Brit Baby Tee: last 7 days sold 12, 9, 7, 11, 11, 4, 11 = 65 units


@pytest.fixture(scope="module")
def base():
    return m.load_base_data()


@pytest.fixture(scope="module")
def sales():
    return m.load_sales()


@pytest.fixture
def tools(base):
    return InventoryTools(base, m.DEFAULT_ASSUMPTIONS)


def row(plan, pid):
    return plan[plan["Product_ID"] == pid].iloc[0]


# ---------------------------------------------------------------- 1. data
def test_data_complete(base, sales):
    assert len(base) == 18
    assert len(sales) == 365                                   # one row per day, no gaps
    assert set(base["Product_ID"]) <= set(sales.columns)       # every product has history
    assert (sales >= 0).all().all()                            # no negative sales
    assert base["Name"].is_unique
    for pid in base["Product_ID"]:
        assert os.path.exists(m.image_path(pid)), f"missing photo for {pid}"


def test_baseline_summary(base):
    s = m.summarize(m.calculate_plan(base))
    assert (s["need_reorder"], s["sufficient_stock"], s["recommended_order_units"]) == (8, 10, 5048)


# ---------------------------------------------------------------- 2. formulas
def test_sma_forecast_by_hand(base, sales):
    assert list(sales[BRIT_TEE].tail(7)) == [12, 9, 7, 11, 11, 4, 11]
    assert row(base, BRIT_TEE)["SMA_7"] == pytest.approx(65 / 7)


def test_brit_tee_full_calculation(base):
    r = row(m.calculate_plan(base), BRIT_TEE)
    assert r["ROP"] == 65                                       # ceil(65/7 * 7)
    assert r["EOQ"] == math.ceil(math.sqrt(2 * (65 / 7) * 365 * 50 / 2))  # 412
    assert r["EOQ"] == 412
    assert r["Current_Stock"] == 65 and r["Status"] == "Reorder"  # stock == ROP -> reorder


def test_annual_cost_by_hand(base, tools):
    # Brit Baby Tee: D = 65/7 x 365 = 3,389.3 units/yr, EOQ = 412
    # ordering = D/EOQ x $50 = $411.32, holding = 412/2 x $2 = $412.00 -> $823.32
    r = row(m.calculate_plan(base), BRIT_TEE)
    assert r["Annual_Cost"] == pytest.approx(823.32, abs=0.01)
    detail = tools.call("get_product_details", {"product_id": BRIT_TEE})
    assert detail["annual_ordering_cost"] == pytest.approx(411.32, abs=0.01)
    assert detail["annual_holding_cost"] == pytest.approx(412.00, abs=0.01)
    # At the EOQ, ordering and holding costs are (almost) equal: EOQ is rounded up.
    assert abs(detail["annual_ordering_cost"] - detail["annual_holding_cost"]) < 1
    assert m.summarize(m.calculate_plan(base))["total_annual_policy_cost"] == pytest.approx(22246.17, abs=0.01)


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
    assert ss == pytest.approx(1.6448536 * 1.25 * 1.0 * 3, rel=1e-6)  # z * 1.25 * MAE * sqrt(9)
    assert m.safety_stock([1.0], 9, 0)[0] == 0


def test_sma_and_mae():
    data = [0, 0, 0, 0, 0, 0, 0, 7]
    assert m.sma_forecast(data, 7) == 1.0
    assert m.sma_mae(data, 7) == 7.0   # only day 8 has a forecast (0); error = 7


def test_window_changes_forecast(base, sales):
    p28 = m.calculate_plan(base, m.validate_assumptions(sma_window=28))
    assert row(p28, BRIT_TEE)["SMA"] == pytest.approx(sales[BRIT_TEE].tail(28).mean())


def test_demand_increase_raises_eoq_and_rop(base):
    before = m.calculate_plan(base)
    after = m.calculate_plan(base, m.validate_assumptions(demand_change_pct=20))
    assert (after["EOQ"] >= before["EOQ"]).all()
    assert (after["ROP"] >= before["ROP"]).all()
    assert m.summarize(after)["recommended_order_units"] == 5962


def test_longer_lead_time(base):
    s = m.summarize(m.calculate_plan(base, m.validate_assumptions(lead_time_days=14)))
    assert s["need_reorder"] == 18   # 14 days > every product's days of cover


# ---------------------------------------------------------------- 3. validation
@pytest.mark.parametrize("bad", [
    {"lead_time_days": -3}, {"lead_time_days": 0}, {"ordering_cost": 0},
    {"holding_cost": "abc"}, {"demand_change_pct": -150}, {"service_level": 80},
    {"sma_window": 5},
])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(m.InputError):
        m.validate_assumptions(**bad)


def test_product_lookup(base):
    plan = m.calculate_plan(base)
    assert m.find_product(plan, "brit tee") == BRIT_TEE
    assert m.find_product(plan, "624486001") == BRIT_TEE          # ID without leading zero
    assert m.find_product(plan, "Brit Baby Tee") == BRIT_TEE
    assert row(plan, m.find_product(plan, "socks"))["Name"] == "7p Basic Shaftless"
    with pytest.raises(m.InputError, match="several"):
        m.find_product(plan, "t-shirt")                           # 3 tees -> asks which
    with pytest.raises(m.InputError):
        m.find_product(plan, "unicorn hat")


# ---------------------------------------------------------------- 4. agent tools
def test_tool_errors_are_returned_not_raised(tools):
    result = tools.call("run_what_if", {"lead_time_days": -5})
    assert "error" in result and result["needs_user_input"]


def test_what_if_does_not_change_dashboard(tools):
    tools.call("run_what_if", {"demand_change_pct": 20})
    assert tools.assumptions == m.DEFAULT_ASSUMPTIONS


def test_update_assumptions_changes_state(tools):
    tools.call("update_assumptions", {"lead_time_days": 14, "sma_window": 14})
    assert (tools.assumptions.lead_time_days, tools.assumptions.sma_window) == (14, 14)


def test_priority_order(tools):
    ranked = tools.call("prioritize_reorders", {"limit": 3})["ranked"]
    assert [r["name"] for r in ranked] == ["Harrison short sleeve top CN", "Milk RW slack", "Lucy blouse"]


def test_offline_asks_for_missing_percentage(tools):
    reply = OfflineAssistant().ask("What happens if demand increases?", [], tools)
    assert "percentage" in reply.text and reply.tool_calls == []


def test_offline_finds_product_by_name(tools):
    reply = OfflineAssistant().ask("Why does Lucy blouse need replenishment?", [], tools)
    assert reply.tool_calls[0][0] == "get_product_details"
    assert "Lucy blouse" in reply.text and "Reorder" in reply.text
