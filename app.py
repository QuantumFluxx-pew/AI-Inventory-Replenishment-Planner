"""
app.py: AI Inventory & Replenishment Planner (Streamlit)

Left panel : inventory dashboard (KPIs, charts, table, product drill-down)
Right panel: Inventory Assistant chat

Files this app uses:
    inventory_model.py   - SMA / EOQ / ROP calculations (the embedded models)
    agent.py             - Gemini tool-calling agent + offline fallback
    inventory_replenishment_results.csv - product forecasts and simulated stock

Run locally:   streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

import inventory_model as model
from agent import DEFAULT_MODEL, GeminiAgent, InventoryTools, OfflineAssistant

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Inventory Replenishment Planner",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================================
# 2. CLEAN WHITE DESIGN
# ==========================================================

st.markdown("""
<style>
    .stApp { background: #f5f7fb; color: #202b43; }
    .block-container { max-width: 1800px; padding-top: 1.5rem;
                       padding-left: 2rem; padding-right: 2rem; }
    h1, h2, h3, h4, p, label { color: #202b43; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e3e8f0; border-radius: 14px;
        padding: 16px; box-shadow: 0 3px 12px rgba(20, 35, 65, 0.03);
    }
    [data-testid="stMetricLabel"] { color: #66758b; }
    [data-testid="stMetricValue"] { color: #202b43; }

    /* Form controls */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: #ffffff; color: #202b43;
        border: 1px solid #dbe3ef; border-radius: 9px;
    }
    div[data-baseweb="select"] *, div[data-baseweb="input"] input { color: #202b43; }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #ffffff; border: 1px solid #e3e8f0;
        border-radius: 12px; padding: 12px;
    }

    /* Buttons */
    .stButton button, .stDownloadButton button, .stFormSubmitButton button {
        background: #ffffff; color: #244b91;
        border: 1px solid #dbe3ef; border-radius: 9px;
    }

    /* Mode badge in the chat header */
    .mode-badge { display: inline-block; padding: 3px 10px; border-radius: 999px;
                  font-size: 0.8rem; font-weight: 600; }
    .mode-llm { background: #e7f6ec; color: #1d6b3a; border: 1px solid #b9e3c7; }
    .mode-offline { background: #fff4e0; color: #8a5a00; border: 1px solid #f1d39b; }

    hr { border-color: #e3e8f0; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. LOAD DATA AND SET UP SESSION STATE
# ==========================================================

@st.cache_data
def load_base():
    # Only Product_ID, SMA_7, MAE and Current_Stock are read from the CSV.
    # EOQ, ROP and Status are recalculated live from the assumptions.
    return model.load_base_data("inventory_replenishment_results.csv")


try:
    base_df = load_base()
except Exception as error:
    st.error("Unable to load inventory data.")
    st.code(str(error))
    st.stop()

# Current model inputs. Shared by the dashboard and the chatbot.
if "assumptions" not in st.session_state:
    st.session_state.assumptions = model.DEFAULT_ASSUMPTIONS

GREETING = (
    "Hello! I'm your inventory replenishment assistant. I can tell you which "
    "products to reorder, how many units to order, why, and what happens "
    "if demand, lead time or costs change.\n\n"
    "For what-if questions I'll ask you for the new value (for example "
    "*demand +20%* or *lead time 14 days*). What would you like to know?"
)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]


def get_secret(name):
    """Read a setting from Streamlit secrets, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # no secrets.toml file, which is normal when running locally
    return os.environ.get(name)


API_KEY = get_secret("GEMINI_API_KEY")
MODEL_NAME = get_secret("GEMINI_MODEL") or DEFAULT_MODEL
AI_ENABLED = bool(API_KEY)

# Recalculate the plan for the current assumptions.
assumptions = st.session_state.assumptions
plan = model.calculate_plan(base_df, assumptions)

# ==========================================================
# 4. HEADER
# ==========================================================

st.title("AI Inventory Replenishment Planner")
st.caption("M5 Sales Forecasting · SMA · EOQ · Reorder Point · AI Assistant")

if assumptions != model.DEFAULT_ASSUMPTIONS:
    st.info(
        "Showing results for **modified assumptions**: "
        f"S = ${assumptions.ordering_cost:g}, H = ${assumptions.holding_cost:g}, "
        f"lead time = {assumptions.lead_time_days:g} days, "
        f"demand change = {assumptions.demand_change_pct:+g}%, "
        f"service level = {assumptions.service_level:g}%. "
        "Use *Reset to defaults* to return to the Milestone 2 baseline."
    )

st.divider()

dashboard_col, chat_col = st.columns([1.8, 1], gap="large")

# ==========================================================
# 5. LEFT PANEL: DASHBOARD
# ==========================================================

with dashboard_col:

    st.header("Inventory Dashboard")

    # ------------------------------------------------------
    # 5a. MODEL INPUTS (user can revise and rerun)
    # ------------------------------------------------------
    with st.expander("⚙️ Model inputs & assumptions", expanded=False):
        st.caption(
            "M5 does not include costs, lead times or stock levels, so these are "
            "labelled assumptions. Change them and click **Recalculate**."
        )
        with st.form("assumptions_form"):
            c1, c2, c3 = st.columns(3)
            ordering_cost = c1.number_input(
                "Ordering cost S ($/order)", value=float(assumptions.ordering_cost), step=5.0)
            holding_cost = c2.number_input(
                "Holding cost H ($/unit/year)", value=float(assumptions.holding_cost), step=0.5)
            lead_time = c3.number_input(
                "Lead time L (days)", value=float(assumptions.lead_time_days), step=1.0)

            c4, c5 = st.columns(2)
            demand_change = c4.number_input(
                "Demand change (%)", value=float(assumptions.demand_change_pct), step=5.0,
                help="Scenario: +20 means every product sells 20% more than the SMA forecast.")
            service_level = c5.selectbox(
                "Safety stock service level",
                model.ALLOWED_SERVICE_LEVELS,
                index=model.ALLOWED_SERVICE_LEVELS.index(assumptions.service_level),
                format_func=lambda v: "None (Milestone 2 model)" if v == 0 else f"{v:g}%",
                help="Safety stock = z × 1.25 × MAE × √L",
            )

            b1, b2 = st.columns(2)
            recalc = b1.form_submit_button("Recalculate", width="stretch")
            reset = b2.form_submit_button("Reset to defaults", width="stretch")

        if recalc:
            try:
                st.session_state.assumptions = model.validate_assumptions(
                    ordering_cost=ordering_cost, holding_cost=holding_cost,
                    lead_time_days=lead_time, demand_change_pct=demand_change,
                    service_level=service_level,
                )
                st.rerun()
            except model.InputError as err:
                # Invalid input: explain and keep the previous values.
                st.error(f"Please fix the inputs: {err}")
        if reset:
            st.session_state.assumptions = model.DEFAULT_ASSUMPTIONS
            st.rerun()

    # ------------------------------------------------------
    # 5b. FILTERS
    # ------------------------------------------------------
    f1, f2 = st.columns(2)
    status_filter = f1.selectbox("Inventory status", ["All", "Reorder", "Sufficient Stock"])
    product_search = f2.text_input("Search product ID", placeholder="HOBBIES_1_001")

    filtered = plan.copy()
    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]
    if product_search:
        filtered = filtered[filtered["Product_ID"].str.contains(
            product_search.strip(), case=False, regex=False)]

    # ------------------------------------------------------
    # 5c. KPI CARDS
    # ------------------------------------------------------
    summary = model.summarize(filtered)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Products Shown", summary["products"])
    k2.metric("Need Reorder", summary["need_reorder"])
    k3.metric("Sufficient Stock", summary["sufficient_stock"])
    k4.metric("Recommended Order Units", f"{summary['recommended_order_units']:,}")

    st.divider()

    # ------------------------------------------------------
    # 5d. CHARTS
    # ------------------------------------------------------
    if filtered.empty:
        st.info("No products match your filters.")
    else:
        st.subheader("Current Stock vs. Reorder Point")
        st.caption("Products whose stock bar is at or below the ROP bar need reordering.")
        st.bar_chart(
            filtered.set_index("Product_ID")[["Current_Stock", "ROP"]],
            stack=False, height=320,
        )

        ch1, ch2 = st.columns(2)
        with ch1:
            st.subheader("Stock Status")
            counts = (filtered["Status"].value_counts()
                      .rename_axis("Status").reset_index(name="Products"))
            st.bar_chart(counts, x="Status", y="Products", height=260)
        with ch2:
            st.subheader("Largest Recommended Orders")
            reorder_rows = filtered[filtered["Status"] == "Reorder"]
            if reorder_rows.empty:
                st.info("No reorder items in this selection.")
            else:
                st.bar_chart(
                    reorder_rows.nlargest(10, "Order_Quantity")
                    .set_index("Product_ID")[["Order_Quantity"]],
                    height=260,
                )

    # ------------------------------------------------------
    # 5e. PRODUCT TABLE + DOWNLOAD
    # ------------------------------------------------------
    st.subheader("Product Inventory Details")
    table_cols = ["Product_ID", "Current_Stock", "ROP", "Safety_Stock", "EOQ",
                  "Daily_Demand", "Annual_Demand", "Days_of_Cover", "Status", "Order_Quantity"]
    st.dataframe(
        filtered[table_cols],
        width="stretch", hide_index=True, height=350,
        column_config={
            "Daily_Demand": st.column_config.NumberColumn("Daily Demand", format="%.2f"),
            "Annual_Demand": st.column_config.NumberColumn("Annual Demand", format="%.1f"),
            "Safety_Stock": st.column_config.NumberColumn("Safety Stock", format="%.2f"),
            "Days_of_Cover": st.column_config.NumberColumn("Days of Cover", format="%.1f"),
        },
    )
    st.download_button(
        "Download Inventory Results",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="inventory_results_filtered.csv",
        mime="text/csv",
    )

    # ------------------------------------------------------
    # 5f. PRODUCT DRILL-DOWN: shows each calculation step
    # ------------------------------------------------------
    st.subheader("How a Recommendation Is Calculated")
    chosen = st.selectbox("Choose a product", plan["Product_ID"].tolist())
    detail = InventoryTools(base_df, assumptions).get_product_details(chosen)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Forecast (units/day)", f"{detail['daily_demand_forecast']:.2f}")
    d2.metric("Reorder Point", detail["reorder_point"])
    d3.metric("EOQ", detail["eoq"])
    d4.metric("Order Now", detail["recommended_order_units"])
    steps = detail["calculation_steps"]
    st.markdown(
        f"1. **Annual demand:** `{steps['annual_demand']}`\n"
        f"2. **EOQ:** `{steps['eoq']}`\n"
        f"3. **Reorder point:** `{steps['reorder_point']}`\n"
        f"4. **Decision:** `{steps['decision_rule']}`\n\n"
        f"Forecast accuracy (MAE of the 7-day SMA): **{detail['forecast_mae']:.3f} units/day**"
    )

    st.caption(
        "Data: M5 Forecasting (Walmart store TX_1, 20 HOBBIES products). "
        "Stock levels are simulated; costs and lead time are assumptions."
    )

# ==========================================================
# 6. RIGHT PANEL: INVENTORY ASSISTANT
# ==========================================================

with chat_col:

    st.header("Inventory Assistant")

    if AI_ENABLED:
        st.markdown(
            f'<span class="mode-badge mode-llm">● AI agent: Gemini ({MODEL_NAME})</span>',
            unsafe_allow_html=True)
        st.caption("The model chooses a tool, Python runs the calculation, "
                   "the model explains the result.")
    else:
        st.markdown(
            '<span class="mode-badge mode-offline">● Offline mode: rule-based, no AI</span>',
            unsafe_allow_html=True)
        st.caption("No GEMINI_API_KEY is configured, so answers come from simple "
                   "keyword rules. Add a key in Streamlit secrets to enable the AI agent.")

    if st.button("Clear conversation"):
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        st.rerun()

    # Conversation, with the tools behind each answer shown for transparency.
    chat_box = st.container(height=560, border=True)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_calls"):
                    with st.expander(f"🔧 Tools used ({len(msg['tool_calls'])})"):
                        for name, args, result in msg["tool_calls"]:
                            st.markdown(f"**{name}**(`{args}`)")
                            st.json(result, expanded=False)
                if msg.get("note"):
                    st.caption(msg["note"])

    # Suggested questions (useful for demos).
    suggestions = ["Which products should I prioritize?",
                   "What if demand increases by 20%?",
                   "Why does HOBBIES_1_019 need replenishment?"]
    picked = None
    s_cols = st.columns(len(suggestions))
    for col, text in zip(s_cols, suggestions):
        if col.button(text, width="stretch"):
            picked = text

    # A form keeps the input inside the right column.
    with st.form("chat_form", clear_on_submit=True):
        typed = st.text_input("Your question", placeholder="Ask about your inventory...")
        sent = st.form_submit_button("Send message", width="stretch")

    question = picked or (typed.strip() if sent else "")

    if question:
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})

        tools = InventoryTools(base_df, st.session_state.assumptions)
        note = None

        with st.spinner("Thinking..."):
            if AI_ENABLED:
                try:
                    reply = GeminiAgent(API_KEY, MODEL_NAME).ask(question, history, tools)
                except Exception as err:
                    # Never fail silently: say Gemini failed and answer offline.
                    reply = OfflineAssistant().ask(question, history, tools)
                    note = (f"Gemini request failed ({type(err).__name__}: {str(err)[:150]}). "
                            "This answer came from offline mode.")
            else:
                reply = OfflineAssistant().ask(question, history, tools)

        # If the agent changed the assumptions, the dashboard follows.
        st.session_state.assumptions = tools.assumptions

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply.text,
            "tool_calls": reply.tool_calls,
            "note": note,
        })
        st.rerun()
