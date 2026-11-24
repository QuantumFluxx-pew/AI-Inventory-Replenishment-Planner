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


# ---------------------------------------------------------------- 5. number check
# Real Gemini answers from the live run-through on 2026-11-11.
from agent import verify_numbers  # noqa: E402

GEMINI_Q4 = ("If demand increases by 20%:\n\nReorders increase: 9 products need reordering (up from 8), "
             "with Charlotte Unpadded Wire newly crossing its threshold.\nOrder quantities & ROPs: Reorder "
             "points rise (e.g., Charlotte Unpadded Wire's ROP goes from 60 to 72 units), and total "
             "recommended order units rise from 5,048 to 5,962 units.\nAnnual policy cost: Rises from "
             "$22,246.17 to $24,369.44 due to larger order sizes.")
GEMINI_Q5 = ("Brit Baby Tee needs replenishment because its current stock of 65 units has fallen right to "
             "its reorder point of 65 units.\nDaily demand forecast (7-day SMA): 9.286 units/day\nLead time: "
             "7 days\nReorder Point (ROP): 9.286 units/day × 7 days = 65 units\nRecommended Action: Place an "
             "order for 412 units (the Economic Order Quantity).")


def _calls(tools, spec):
    return [(name, args, tools.call(name, args)) for name, args in spec]


def test_real_gemini_answers_pass(tools, base):
    names = list(base["Name"])
    q4 = _calls(tools, [("run_what_if", {"demand_change_pct": 20})])
    q5 = _calls(tools, [("get_product_details", {"product_id": "Brit Baby Tee"})])
    r4 = verify_numbers(GEMINI_Q4, q4, "What if demand increases by 20%?", [], names)
    r5 = verify_numbers(GEMINI_Q5, q5, "Why does Brit Baby Tee need replenishment?", [], names)
    assert r4["unmatched"] == [] and r4["checked"] >= 9
    assert r5["unmatched"] == [] and r5["checked"] >= 8


def test_invented_numbers_are_caught(tools, base):
    q5 = _calls(tools, [("get_product_details", {"product_id": "Brit Baby Tee"})])
    r = verify_numbers("Order 950 units; it has 2.7 days of cover.", q5, "", [], list(base["Name"]))
    assert r["unmatched"] == ["950", "2.7"]
    # A number with no calculation behind it at all is caught too.
    assert verify_numbers("Order 500 units.", [], "", [], [])["unmatched"] == ["500"]


def test_check_allows_question_examples_and_differences(tools, base):
    names = list(base["Name"])
    q3 = _calls(tools, [("update_assumptions", {"lead_time_days": -5})])
    r = verify_numbers("Lead time must be greater than 0 days, so -5 days is not valid "
                       "(for example, 10 or 14 days).", q3, "Set lead time to -5 days", [], names)
    assert r["unmatched"] == [] and r["examples"] == ["10", "14"]
    q4 = _calls(tools, [("run_what_if", {"demand_change_pct": 20})])
    assert verify_numbers("Units rise by 914.", q4, "", [], names)["unmatched"] == []  # 5,962 - 5,048
    assert verify_numbers("1. **Lucy blouse** is urgent.", q4, "", [], names) is None  # list marker only


def test_offline_demo_answers_pass(base):
    tools = InventoryTools(base, m.DEFAULT_ASSUMPTIONS)
    names, earlier = list(base["Name"]), []
    for q in ["Which products should I prioritize?", "What if demand increases by 20%?",
              "Why does Lucy blouse need replenishment?", "What if lead time is 14 days?"]:
        reply = OfflineAssistant().ask(q, [], tools)
        r = verify_numbers(reply.text, reply.tool_calls, q, earlier, names)
        assert r and r["unmatched"] == [], (q, r)
        earlier += reply.tool_calls


def test_agent_retries_empty_reply(tools):
    """If Gemini answers with no words (e.g. just "-"), the agent asks once more."""
    from google.genai import types
    from agent import GeminiAgent

    class Resp:
        def __init__(self, parts):
            self.candidates = [types.Candidate(content=types.Content(role="model", parts=parts))]

        @property
        def function_calls(self):
            return [p.function_call for p in self.candidates[0].content.parts if p.function_call] or None

        @property
        def text(self):
            return "".join(p.text or "" for p in self.candidates[0].content.parts)

    script = [[types.Part(function_call=types.FunctionCall(name="get_inventory_summary", args={}))],
              [types.Part(text="-")],
              [types.Part(text="You have 3,346 units in stock across 18 products.")]]

    class Models:
        calls = 0

        def generate_content(self, model, contents, config):
            Models.calls += 1
            return Resp(script[Models.calls - 1])

    class Client:
        models = Models()

    reply = GeminiAgent("test-key", client=Client()).ask("how many units available?", [], tools)
    assert reply.text.startswith("You have 3,346 units")
    assert Models.calls == 3


# ---------------------------------------------------------------- 6. second forecast model
def test_ses_by_hand():
    # Level starts at the 7-day average (10). Day 8 sells 20, so with alpha 0.5
    # the level moves halfway: 0.5 x 20 + 0.5 x 10 = 15 = tomorrow's forecast.
    data = [10] * 7 + [20]
    assert list(m.ses_backtest(data, 0.5)) == [10.0] * 8
    assert m.ses_forecast(data, 0.5) == 15.0
    # alpha = 1 means "tomorrow = today".
    assert m.ses_forecast([3, 5, 9, 4, 6, 8, 2, 7], 1.0) == 7.0
    with pytest.raises(m.InputError):
        m.ses_backtest(data, 0)


def test_ses_alpha_fitted_per_product(base):
    assert set(base["SES_ALPHA"]) <= set(m.SES_ALPHAS)
    r = row(base, BRIT_TEE)
    assert 0 < r["SES"] < 50 and r["HOLD_SES"] > 0


def test_best_model_picks_lower_holdout_error(base):
    plan = m.calculate_plan(base, m.validate_assumptions(forecast_model="best"))
    expected = plan[["HOLD_SES", "HOLD_SMA_7"]].min(axis=1)
    assert (plan["Holdout_MAE"] == expected).all()
    # On the hold-out period, exponential smoothing beats the 7-day average
    # for 14 of the 18 products.
    assert int((base["HOLD_SES"] < base["HOLD_SMA_7"]).sum()) == 14
    # The default is still the 7-day moving average (Milestone 2 results unchanged).
    default = m.calculate_plan(base)
    assert default["Forecast_Model"].eq("7-day moving average").all()


def test_forecast_model_validation():
    assert m.validate_assumptions(forecast_model="Exponential smoothing").forecast_model == "ses"
    with pytest.raises(m.InputError, match="Forecast model"):
        m.validate_assumptions(forecast_model="arima")


def test_agent_can_switch_forecast_model(tools):
    tools.call("update_assumptions", {"forecast_model": "best"})
    assert tools.assumptions.forecast_model == "best"
    detail = tools.call("get_product_details", {"product_id": "Harrison"})
    cmp = detail["model_comparison_on_holdout"]
    assert cmp["more_accurate_model"] == "exponential smoothing"
    assert detail["forecast_model"].startswith("Exponential smoothing")


# ---------------------------------------------------------------- 7. user-uploaded data
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "example_upload")


def _example():
    return (pd.read_csv(os.path.join(EXAMPLE_DIR, "example_sales.csv")),
            pd.read_csv(os.path.join(EXAMPLE_DIR, "example_stock.csv")))


def test_upload_gives_same_results_as_sample():
    sales, stock = _example()
    base, wide, notes = m.prepare_uploaded_data(sales, stock)
    plan = m.calculate_plan(base).set_index("Name")
    assert len(wide) == 120 and len(plan) == 5 and notes == []
    # Same sales and stock as the H&M sample -> same orders.
    assert plan.loc["Brit Baby Tee", "Order_Quantity"] == 412
    assert plan.loc["Harrison short sleeve top CN", "Order_Quantity"] == 704


def test_upload_without_stock_estimates_it():
    sales, _ = _example()
    base, _, notes = m.prepare_uploaded_data(sales)
    assert "estimated" in notes[0]
    assert (base["Current_Stock"] > 0).all()


@pytest.mark.parametrize("change, message", [
    (lambda s: s.rename(columns={"units": "price"}), "units"),
    (lambda s: s.assign(units=s["units"].astype(object).where(s.index != 3, "lots")), "not a number"),
    (lambda s: s.assign(units=s["units"].where(s.index != 5, -4)), "negative"),
    (lambda s: s.assign(date=s["date"].where(s.index != 7, "yesterday")), "date"),
    (lambda s: s[s["date"] >= "2020-08-01"], "at least 60 days"),
])
def test_upload_rejects_bad_files(change, message):
    sales, stock = _example()
    with pytest.raises(m.InputError, match=message):
        m.prepare_uploaded_data(change(sales), stock)


def test_upload_rejects_missing_stock_rows():
    sales, stock = _example()
    with pytest.raises(m.InputError, match="no stock for"):
        m.prepare_uploaded_data(sales, stock.iloc[:3])


def test_agent_retries_when_gemini_is_busy(tools):
    """A brief 503 'model overloaded' error is retried instead of failing."""
    from google.genai import types
    from agent import GeminiAgent

    class Busy(Exception):
        code = 503

    class Resp:
        candidates = [types.Candidate(content=types.Content(role="model",
                      parts=[types.Part(text="You have 3,346 units in stock across 18 products.")]))]
        function_calls = None
        text = "You have 3,346 units in stock across 18 products."

    class Models:
        calls = 0

        def generate_content(self, model, contents, config):
            Models.calls += 1
            if Models.calls == 1:
                raise Busy("503 UNAVAILABLE")
            return Resp()

    class Client:
        models = Models()

    agent = GeminiAgent("test-key", client=Client())
    agent.retry_waits = (0, 0)   # no real waiting in tests
    reply = agent.ask("how many units available?", [], tools)
    assert reply.text.startswith("You have 3,346") and Models.calls == 2


# ---------------------------------------------------------------- 8. dashboard extras
def test_units_at_risk_by_hand(base):
    # Harrison top: 27.14/day x 7 days = 190 units needed, 97 in stock -> 93 at risk.
    plan = m.calculate_plan(base)
    assert row(plan, "0678942001")["Units_At_Risk"] == 93
    # Brit Baby Tee lasts exactly the 7-day lead time -> nothing at risk.
    assert row(plan, BRIT_TEE)["Units_At_Risk"] == 0


def test_cost_curve_lowest_at_eoq():
    # Brit Baby Tee: D = 3,389.3, S = $50, H = $2 -> EOQ about 411.7, total about $823.
    curve = m.cost_curve(65 / 7 * 365, 50, 2, points=281)
    best = curve.loc[curve["Total_Cost"].idxmin()]
    assert best["Order_Size"] == pytest.approx(411.7, rel=0.02)
    assert best["Total_Cost"] == pytest.approx(823.3, abs=1)
    # Ordering cost falls and holding cost rises as orders get bigger.
    assert curve["Ordering_Cost"].is_monotonic_decreasing
    assert curve["Holding_Cost"].is_monotonic_increasing
    with pytest.raises(m.InputError):
        m.cost_curve(0, 50, 2)


def test_safety_stock_tradeoff(base):
    trade = m.safety_stock_tradeoff(base, m.DEFAULT_ASSUMPTIONS)
    assert list(trade["Service_Level"]) == list(m.ALLOWED_SERVICE_LEVELS)
    assert trade.iloc[0]["Safety_Stock_Units"] == 0            # none = no safety stock
    assert trade["Extra_Holding_Cost"].is_monotonic_increasing  # more safety costs more
    # Extra cost = safety stock units x $2 holding cost.
    r = trade[trade["Service_Level"] == 95].iloc[0]
    assert r["Extra_Holding_Cost"] == pytest.approx(r["Safety_Stock_Units"] * 2, abs=1)


def test_voice_question_is_transcribed():
    """A spoken question becomes plain text, then goes through the normal agent."""
    from agent import transcribe_audio

    class Resp:
        text = '  "Which products should I   prioritize?" \n'

    class Models:
        def generate_content(self, model, contents, config):
            assert contents[0].inline_data.mime_type == "audio/wav"
            assert "exactly what the speaker says" in contents[1]
            return Resp()

    class Client:
        models = Models()

    assert transcribe_audio(b"RIFF....", "key", client=Client()) == "Which products should I prioritize?"
    assert transcribe_audio(b"", "key", client=Client()) == ""
