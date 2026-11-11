"""
app.py: AI Inventory & Replenishment Planner (Streamlit)

Left panel : inventory dashboard (KPIs, charts, table, product drill-down)
Right panel: Stock Pilot (AI inventory agent chat)

Files this app uses:
    inventory_model.py   - SMA / MAE / EOQ / ROP calculations (the embedded models)
    agent.py             - Gemini tool-calling agent + offline fallback
    data/products.csv    - 18 H&M products, with simulated current stock
    data/daily_sales.csv - one year of real H&M daily unit sales
    images/<id>.jpg      - product photo thumbnails

Run locally:   streamlit run app.py
"""

import base64
import importlib
import os

import altair as alt
import pandas as pd
import streamlit as st

import agent as agent_module
import inventory_model as model

# Streamlit Cloud reloads app.py after a deploy but can keep older copies of
# the helper modules in memory, which breaks the app. Reload them if stale.
if getattr(model, "MODEL_VERSION", None) != getattr(agent_module, "MODEL_VERSION", None) \
        or not hasattr(model, "ALLOWED_SMA_WINDOWS"):
    model = importlib.reload(model)
    agent_module = importlib.reload(agent_module)

DEFAULT_MODEL = agent_module.DEFAULT_MODEL
GeminiAgent = agent_module.GeminiAgent
InventoryTools = agent_module.InventoryTools
OfflineAssistant = agent_module.OfflineAssistant

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Inventory Replenishment Planner",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================================
# 2. CLEAN WHITE DESIGN
# ==========================================================

# Colors come from .streamlit/config.toml (light theme, red accent).
# This CSS adds a minimal fashion-retail look: square edges, thin black
# lines, uppercase letter-spaced labels and generous white space.
# (H&M-inspired style only: no H&M logo or trademarked artwork is used.)
st.markdown("""
<style>
    /* Fonts: Playfair Display (elegant serif) for the main title,
       Montserrat (bold, modern retail style) for section headings,
       DM Sans (clean modern sans) for everything else. */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Montserrat:wght@600;700;800&family=Syne:wght@700;800&family=Space+Grotesk:wght@600;700&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&display=swap');
    html, body, .stApp, .stMarkdown, .stCaption, p, li, label, input, textarea, button,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-baseweb="tab"] {
        font-family: 'DM Sans', 'Helvetica Neue', Arial, sans-serif !important;
    }
    h1, h2, h3 {
        font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif !important;
        font-weight: 800; letter-spacing: 0.02em; color: #111;
    }
    .block-container { max-width: 1800px; padding-top: 3.2rem;
                       padding-left: 2rem; padding-right: 2rem; }
    h2 { font-size: 1.45rem; text-transform: uppercase; letter-spacing: 0.06em; }

    /* Panel headings: each side has its own typeface */
    h2.panel-title { font-size: 2.1rem; line-height: 1.15; margin: 0.4rem 0 0.8rem 0;
                     text-transform: uppercase; color: #111; }
    h2.dash-title { font-family: 'Syne', 'Helvetica Neue', Arial, sans-serif !important;
                    font-weight: 800; letter-spacing: 0.03em; }
    h2.assist-title { font-family: 'Space Grotesk', 'Helvetica Neue', Arial, sans-serif !important;
                      font-weight: 700; letter-spacing: 0.01em; }

    /* Stock Pilot chat panel */
    .advisor-sub { color: #555; margin: -0.5rem 0 0.6rem 0; font-size: 0.95rem; }
    .st-key-advisor_chat { background: #faf9f6; border: 1px solid #111 !important;
                           border-radius: 0 !important; }
    [data-testid="stChatMessage"] { background: transparent; border: none; padding: 6px 2px;
                                    gap: 10px; border-radius: 0; }
    [data-testid="stChatMessageContent"] { background: #fff; border: 1px solid #e5e5e5;
                                           padding: 10px 14px; min-width: 0; overflow-wrap: anywhere; }
    /* Wide answers (tables) scroll inside their own bubble, never the whole chat. */
    .st-key-advisor_chat, .st-key-advisor_chat > div { overflow-x: hidden !important; }
    [data-testid="stChatMessageContent"] table { display: block; max-width: 100%; overflow-x: auto;
                                                 font-size: 0.85rem; }
    [data-testid="stChatMessageContent"] th, [data-testid="stChatMessageContent"] td {
        padding: 4px 8px; overflow-wrap: normal; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"] { background: #111; border-color: #111;
                                               flex: 0 1 auto; max-width: 85%; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"] p { color: #fff; }
    [data-testid="stChatMessageAvatarAssistant"] { background: #111 !important; color: #fff !important;
                                                   border-radius: 0 !important; }
    [data-testid="stChatMessageAvatarUser"] { background: #d0021b !important; color: #fff !important;
                                              border-radius: 0 !important; }
    [class*="st-key-suggest_"] button, [class*="st-key-suggest_"] button p {
        text-transform: none !important; letter-spacing: 0 !important;
        font-weight: 500 !important; font-size: 0.85rem !important; }
    .try-label { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
                 color: #555; margin: 0.6rem 0 0.2rem 0; }
    .stFormSubmitButton button[kind="primaryFormSubmit"] { background: #111; color: #fff; border-color: #111; }
    .stFormSubmitButton button[kind="primaryFormSubmit"]:hover { background: #d0021b; border-color: #d0021b; }

    /* Header band */
    .hero { background: #111; color: #fff; padding: 22px 28px; margin-bottom: 18px;
            border-bottom: 4px solid #d0021b; }
    .hero .eyebrow { font-size: 0.72rem; letter-spacing: 0.22em; text-transform: uppercase;
                     color: #bbb; margin: 0 0 6px 0; }
    .hero h1 { font-family: 'Playfair Display', Georgia, serif !important; color: #fff !important; font-size: 2.4rem; margin: 0; padding: 0;
               text-transform: none; font-weight: 700; letter-spacing: 0.01em; }
    .hero p { color: #ddd; margin: 6px 0 0 0; font-size: 0.92rem; }

    /* Metric cards: flat, square, thin border */
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #111; border-radius: 0; padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] p { color: #555; text-transform: uppercase;
                                      letter-spacing: 0.08em; font-size: 0.72rem; }
    [data-testid="stMetricValue"] { font-weight: 700; }

    /* Tabs: uppercase, letter-spaced */
    button[data-baseweb="tab"] p { text-transform: uppercase; letter-spacing: 0.08em;
                                   font-size: 0.8rem; font-weight: 600; }

    /* Form controls and buttons: square edges */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { border-radius: 0; }
    .stButton button, .stDownloadButton button, .stFormSubmitButton button {
        background: #fff; color: #111; border: 1px solid #111; border-radius: 0;
        text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.78rem; font-weight: 600;
    }
    .stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
        background: #111; color: #fff; border-color: #111;
    }

    /* Product cards (reorder strip) */
    .pcard { border: 1px solid #e5e5e5; background: #fff; padding: 8px; height: 100%; }
    .pcard img { width: 100%; aspect-ratio: 2 / 3; object-fit: cover; background: #f6f6f4; }
    .pcard .pname { font-weight: 600; font-size: 0.85rem; margin-top: 6px; line-height: 1.2; }
    .pcard .ptype { color: #666; font-size: 0.75rem; }
    .pcard .porder { color: #d0021b; font-weight: 700; font-size: 0.85rem; margin-top: 4px; }
    .pcard .pcover { color: #444; font-size: 0.75rem; }

    /* Chat messages */
    [data-testid="stChatMessage"] { background: #fff; border: 1px solid #e5e5e5;
                                    border-radius: 0; padding: 12px; }

    /* Mode badge in the chat header */
    .mode-badge { display: inline-block; padding: 3px 10px; border-radius: 0;
                  font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; }
    .mode-llm { background: #e7f6ec; color: #1d6b3a; border: 1px solid #1d6b3a; }
    .mode-offline { background: #fff4e0; color: #8a5a00; border: 1px solid #8a5a00; }

    hr { border-color: #e5e5e5; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. LOAD DATA AND SET UP SESSION STATE
# ==========================================================

@st.cache_data
def load_data(version):
    """Products + forecasts (SMA and MAE for every window) and daily sales.
    `version` is only here so a new deploy never reuses an old cache."""
    return model.load_base_data(), model.load_sales()


@st.cache_data
def photo_uri(product_id):
    """Product photo as a data URI, so tables and chat can show it."""
    try:
        with open(model.image_path(product_id), "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return None


try:
    base_df, sales_df = load_data(model.MODEL_VERSION)
except Exception as error:
    st.error("Unable to load inventory data.")
    st.code(str(error))
    st.stop()

# Current model inputs. Shared by the dashboard and the chatbot.
# Reset them if they come from an older version of the app.
if not isinstance(st.session_state.get("assumptions"), model.Assumptions):
    st.session_state.assumptions = model.DEFAULT_ASSUMPTIONS

GREETING = (
    "Hello! I'm Stock Pilot, your AI inventory agent for 18 H&M products, "
    "from basics like the *Brit Baby Tee* to the *Pluto RW slacks*. I can tell "
    "you which products to reorder, how many units to order, why, and what "
    "happens if demand, lead time, costs or the forecast window change.\n\n"
    "For what-if questions I'll ask you for the new value (for example "
    "*demand +20%* or *lead time 14 days*). What would you like to know?"
)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
# Keep the greeting current if the app was updated while this chat was open.
if (st.session_state.messages and st.session_state.messages[0]["role"] == "assistant"
        and st.session_state.messages[0]["content"] != GREETING
        and st.session_state.messages[0]["content"].startswith("Hello! I'm")):
    st.session_state.messages[0] = {"role": "assistant", "content": GREETING}


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
# 3b. CHARTS (Altair, built into Streamlit)
# ==========================================================
# Status colors are reserved for status only: red = Reorder, green = OK.
# Every chart also has a legend or label, so meaning never relies on color alone.

STATUS_COLORS = alt.Scale(domain=["Reorder", "Sufficient Stock"],
                          range=["#d03b3b", "#0ca30c"])
INK = "#52514e"        # neutral dark gray for reference marks (ROP, lead time)
ORDER_BLUE = "#111111"  # single-series color for order quantities (black, retail style)
GRID = "#e1e0d9"


def _style(chart, height):
    """Shared look: light gray grid, readable labels, no chart border."""
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domainColor="#c3c2b7",
                            labelColor="#202b43", titleColor="#52514e",
                            labelFontSize=12, titleFontSize=12, labelLimit=260)
            .configure_legend(orient="top", title=None, labelFontSize=12))


def stock_vs_rop_chart(data):
    """Horizontal bars = current stock (colored by status); dark tick = ROP.
    A bar that ends at or before its tick needs reordering."""
    base = alt.Chart(data).encode(
        y=alt.Y("Name:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False)),
        tooltip=[
            alt.Tooltip("Name", title="Product"), alt.Tooltip("Product_ID", title="H&M article ID"),
            alt.Tooltip("Current_Stock", title="Current stock"),
            alt.Tooltip("ROP", title="Reorder point"),
            alt.Tooltip("Status"),
            alt.Tooltip("Order_Quantity", title="Order now"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X("Current_Stock:Q", title="Units"),
        color=alt.Color("Status:N", scale=STATUS_COLORS),
    )
    rop_tick = base.mark_tick(color=INK, thickness=3, size=18).encode(x="ROP:Q")
    return _style(bars + rop_tick, height=max(160, 24 * len(data)))


def days_of_cover_chart(data, lead_time):
    """How many days current stock lasts at forecast demand, compared with
    the supplier lead time (dashed line). Bars left of the line run out
    before a new order could arrive."""
    covered = data.dropna(subset=["Days_of_Cover"]).sort_values("Days_of_Cover")
    base = alt.Chart(covered).encode(
        y=alt.Y("Name:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False)))
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X("Days_of_Cover:Q", title="Days of stock left at forecast demand"),
        color=alt.Color("Status:N", scale=STATUS_COLORS),
        tooltip=[alt.Tooltip("Name", title="Product"), alt.Tooltip("Product_ID", title="H&M article ID"),
                 alt.Tooltip("Days_of_Cover", title="Days of cover"),
                 alt.Tooltip("Status")],
    )
    line_data = pd.DataFrame({"x": [lead_time],
                              "label": [f"Lead time: {lead_time:g} days"]})
    rule = alt.Chart(line_data).mark_rule(color=INK, strokeDash=[6, 4],
                                          strokeWidth=2).encode(x="x:Q")
    label = alt.Chart(line_data).mark_text(
        align="left", dx=6, dy=-6, color=INK, fontSize=12, fontWeight="bold"
    ).encode(x="x:Q", y=alt.value(0), text="label:N")
    return _style(bars + rule + label, height=max(160, 24 * len(covered)))


def order_quantity_chart(data):
    """Recommended order units, largest first, with the value at each bar end."""
    top = data[data["Order_Quantity"] > 0].nlargest(10, "Order_Quantity")
    base = alt.Chart(top).encode(
        y=alt.Y("Name:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)))
    bars = base.mark_bar(color=ORDER_BLUE, cornerRadiusEnd=4,
                         height={"band": 0.7}).encode(
        x=alt.X("Order_Quantity:Q", title="Units to order (EOQ)",
                axis=alt.Axis(tickCount=6)),
        tooltip=[alt.Tooltip("Name", title="Product"), alt.Tooltip("Product_ID", title="H&M article ID"),
                 alt.Tooltip("Order_Quantity", title="Units to order")],
    )
    values = base.mark_text(align="left", dx=4, color="#202b43", fontSize=12).encode(
        x="Order_Quantity:Q", text="Order_Quantity:Q")
    return _style(bars + values, height=max(120, 28 * len(top)))


def sales_history_chart(daily_sales, window, days=90):
    """Actual daily units sold (bars) vs. the one-day-ahead moving-average
    forecast (line) for the last `days` days. Shows how well the SMA tracks."""
    forecast = model.sma_backtest(daily_sales.reset_index(drop=True), window).values
    frame = pd.DataFrame({"Date": daily_sales.index, "Actual units sold": daily_sales.values,
                          f"{window}-day SMA forecast": forecast}).tail(days)
    long = frame.melt("Date", var_name="Series", value_name="Units")
    colors = alt.Scale(domain=["Actual units sold", f"{window}-day SMA forecast"],
                       range=["#cfcfcb", "#111111"])
    tips = [alt.Tooltip("Date:T"), alt.Tooltip("Series:N"), alt.Tooltip("Units:Q", format=".1f")]
    bars = alt.Chart(long[long["Series"] == "Actual units sold"]).mark_bar(width={"band": 0.8}).encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Units:Q", title="Units per day"),
        color=alt.Color("Series:N", scale=colors), tooltip=tips)
    line = alt.Chart(long[long["Series"] != "Actual units sold"]).mark_line(strokeWidth=2.5).encode(
        x="Date:T", y="Units:Q", color=alt.Color("Series:N", scale=colors), tooltip=tips)
    return _style(bars + line, height=240)

# ==========================================================
# 4. HEADER
# ==========================================================

st.markdown(
    '<div class="hero"><h1>AI Inventory Replenishment Planner</h1></div>',
    unsafe_allow_html=True,
)

if assumptions != model.DEFAULT_ASSUMPTIONS:
    st.info(
        "Showing results for **modified assumptions**: "
        f"S = ${assumptions.ordering_cost:g}, H = ${assumptions.holding_cost:g}, "
        f"lead time = {assumptions.lead_time_days:g} days, "
        f"demand change = {assumptions.demand_change_pct:+g}%, "
        f"service level = {assumptions.service_level:g}%, "
        f"forecast window = {assumptions.sma_window} days. "
        "Use *Reset to defaults* to return to the baseline."
    )

dashboard_col, chat_col = st.columns([1.8, 1], gap="large")

# ==========================================================
# 5. LEFT PANEL: DASHBOARD
# ==========================================================

with dashboard_col:

    st.markdown('<h2 class="panel-title dash-title">Inventory Dashboard</h2>', unsafe_allow_html=True)

    # ------------------------------------------------------
    # 5a. FILTERS + KPI CARDS (shared by all tabs)
    # ------------------------------------------------------
    f1, f2 = st.columns(2)
    status_filter = f1.selectbox("Inventory status", ["All", "Reorder", "Sufficient Stock"])
    product_search = f2.text_input("Search product", placeholder="e.g. tee, slacks, socks")

    filtered = plan.copy()
    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]
    if product_search:
        # Match the product name, its type (e.g. 'socks') or its H&M article ID.
        term = product_search.strip()
        filtered = filtered[
            filtered["Name"].str.contains(term, case=False, regex=False)
            | filtered["Product_Type"].str.contains(term, case=False, regex=False)
            | filtered["Product_ID"].str.contains(term, case=False, regex=False)]

    summary = model.summarize(filtered)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Products Shown", summary["products"])
    k2.metric("Need Reorder", summary["need_reorder"])
    k3.metric("Sufficient Stock", summary["sufficient_stock"])
    k4.metric("Units to Order", f"{summary['recommended_order_units']:,}")

    tab_overview, tab_products, tab_details, tab_settings = st.tabs(
        ["Overview", "Products", "Product Details", "Settings"])

    # ------------------------------------------------------
    # 5b. OVERVIEW TAB: charts
    # ------------------------------------------------------
    with tab_overview:
        if filtered.empty:
            st.info("No products match your filters.")
        else:
            # "Reorder now" strip: product photos, most urgent first.
            urgent = model.prioritize(filtered).head(6)
            if not urgent.empty:
                st.markdown("**REORDER NOW** · most urgent first")
                cards = st.columns(6)
                for col, (_, r) in zip(cards, urgent.iterrows()):
                    img = photo_uri(r["Product_ID"]) or ""
                    col.markdown(
                        f'<div class="pcard"><img src="{img}" alt="{r["Name"]}">'
                        f'<div class="pname">{r["Name"]}</div>'
                        f'<div class="ptype">{r["Product_Type"]} · {r["Colour"]}</div>'
                        f'<div class="porder">Order {int(r["Order_Quantity"]):,}</div>'
                        f'<div class="pcover">{r["Days_of_Cover"]:g} days of stock left</div></div>',
                        unsafe_allow_html=True)
                st.write("")

            view = st.segmented_control(
                "Chart", ["Stock vs. Reorder Point", "Days of Cover", "Recommended Orders"],
                default="Stock vs. Reorder Point", label_visibility="collapsed",
                key="chart_view") or "Stock vs. Reorder Point"

            if view == "Stock vs. Reorder Point":
                st.caption("Bar = current stock · dark tick = reorder point (ROP). "
                           "A bar that ends at or before its tick needs reordering.")
                chart = stock_vs_rop_chart(filtered)
            elif view == "Days of Cover":
                st.caption("How long current stock lasts at forecast demand. Products left "
                           "of the dashed line run out before a new order could arrive. "
                           "Products with zero forecast demand are not shown.")
                chart = days_of_cover_chart(filtered, assumptions.lead_time_days)
            else:
                chart = None
                if (filtered["Order_Quantity"] > 0).any():
                    st.caption("Largest recommended orders (EOQ), up to 10 products.")
                    chart = order_quantity_chart(filtered)
                else:
                    st.info("No products in this selection need an order.")
            if chart is not None:
                st.altair_chart(chart, width="stretch", height=chart.height)

    # ------------------------------------------------------
    # 5c. PRODUCTS TAB: table with photos + download
    # ------------------------------------------------------
    with tab_products:
        # Status shown as a text label, so meaning never depends on color.
        table = filtered.copy()
        table["Status"] = table["Status"].map(
            {"Reorder": "Reorder", "Sufficient Stock": "Sufficient"})
        table["Photo"] = table["Product_ID"].map(photo_uri)
        table_cols = ["Photo", "Name", "Product_Type", "Status", "Current_Stock", "ROP",
                      "Order_Quantity", "EOQ", "Days_of_Cover", "Daily_Demand", "MAE",
                      "Annual_Demand", "Safety_Stock", "Product_ID"]
        st.dataframe(
            table[table_cols],
            width="stretch", hide_index=True, height=560, row_height=56,
            column_config={
                "Photo": st.column_config.ImageColumn("Photo", width="small"),
                "Name": st.column_config.TextColumn("Product"),
                "Product_Type": st.column_config.TextColumn("Type"),
                "Product_ID": st.column_config.TextColumn("H&M Article ID"),
                "MAE": st.column_config.NumberColumn(
                    "Forecast MAE", format="%.2f",
                    help="Average forecast error of the moving average, units/day (back-tested over the year)"),
                "Current_Stock": st.column_config.NumberColumn("Stock", help="Units on hand (simulated)"),
                "ROP": st.column_config.NumberColumn("Reorder Point", help="Reorder when stock ≤ this"),
                "Order_Quantity": st.column_config.NumberColumn("Order Now", help="Units to order (EOQ if reordering)"),
                "EOQ": st.column_config.NumberColumn("EOQ", help="Economic order quantity"),
                "Days_of_Cover": st.column_config.NumberColumn("Days of Cover", format="%.1f",
                                                               help="Stock ÷ forecast daily demand"),
                "Daily_Demand": st.column_config.NumberColumn("Daily Demand", format="%.2f"),
                "Annual_Demand": st.column_config.NumberColumn("Annual Demand", format="%.0f"),
                "Safety_Stock": st.column_config.NumberColumn("Safety Stock", format="%.1f"),
            },
        )
        download_cols = ["Product_ID", "Name", "Product_Type", "Colour", "Current_Stock", "SMA",
                         "MAE", "Daily_Demand", "Annual_Demand", "EOQ", "Safety_Stock", "ROP",
                         "Days_of_Cover", "Status", "Order_Quantity", "Annual_Cost"]
        st.download_button(
            "Download Inventory Results",
            data=filtered[download_cols].to_csv(index=False).encode("utf-8"),
            file_name="inventory_results_filtered.csv",
            mime="text/csv",
        )

    # ------------------------------------------------------
    # 5d. PRODUCT DETAILS TAB: each calculation step
    # ------------------------------------------------------
    with tab_details:
        names = dict(zip(plan["Product_ID"], plan["Name"]))
        chosen = st.selectbox("Choose a product", plan["Product_ID"].tolist(),
                              format_func=lambda pid: names[pid])
        detail = InventoryTools(base_df, assumptions).get_product_details(chosen)

        pic_col, info_col = st.columns([1, 3])
        with pic_col:
            if os.path.exists(model.image_path(chosen)):
                st.image(model.image_path(chosen), width="stretch")
        with info_col:
            st.markdown(f"**{detail['name']}** · {detail['product_type']} · {detail['colour']}")
            if detail["description"]:
                st.caption(detail["description"])
            st.caption(f"H&M article ID {detail['product_id']}")
            st.markdown(f"**{detail['status']}** · stock {detail['current_stock']} units · "
                        f"{detail['days_of_cover']} days of cover")

        # Full-width row so the numbers are never cut off.
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Forecast (units/day)", f"{detail['daily_demand_forecast']:.2f}")
        d2.metric("Reorder Point", detail["reorder_point"])
        d3.metric("EOQ", detail["eoq"])
        d4.metric("Order Now", detail["recommended_order_units"])

        steps = detail["calculation_steps"]
        st.markdown(
            f"1. **Forecast:** average of the last {detail['sma_window_days']} days of sales = "
            f"`{detail['sma_forecast_units_per_day']} units/day`\n"
            f"2. **Annual demand:** `{steps['annual_demand']}`\n"
            f"3. **EOQ:** `{steps['eoq']}`\n"
            f"4. **Reorder point:** `{steps['reorder_point']}`\n"
            f"5. **Decision:** `{steps['decision_rule']}`"
        )

        st.markdown(f"**Sales history vs. forecast (last 90 days)**: forecast accuracy "
                    f"(MAE over the year) = **{detail['forecast_mae_units_per_day']:.2f} units/day**")
        chart = sales_history_chart(sales_df[chosen], assumptions.sma_window)
        st.altair_chart(chart, width="stretch", height=chart.height)

    # ------------------------------------------------------
    # 5e. SETTINGS TAB: model inputs (user can revise and rerun)
    # ------------------------------------------------------
    with tab_settings:
        st.caption(
            "The H&M data has real sales but no costs, lead times or stock levels, "
            "so these are labelled assumptions. Change them and click **Recalculate**; "
            "every tab and the assistant use the new values."
        )
        with st.form("assumptions_form"):
            c1, c2, c3 = st.columns(3)
            ordering_cost = c1.number_input(
                "Ordering cost S ($/order)", value=float(assumptions.ordering_cost), step=5.0)
            holding_cost = c2.number_input(
                "Holding cost H ($/unit/year)", value=float(assumptions.holding_cost), step=0.5)
            lead_time = c3.number_input(
                "Lead time L (days)", value=float(assumptions.lead_time_days), step=1.0)

            c4, c5, c6 = st.columns(3)
            demand_change = c4.number_input(
                "Demand change (%)", value=float(assumptions.demand_change_pct), step=5.0,
                help="Scenario: +20 means every product sells 20% more than the SMA forecast.")
            service_level = c5.selectbox(
                "Safety stock service level",
                model.ALLOWED_SERVICE_LEVELS,
                index=model.ALLOWED_SERVICE_LEVELS.index(assumptions.service_level),
                format_func=lambda v: "None" if v == 0 else f"{v:g}%",
                help="Safety stock = z × 1.25 × MAE × √L",
            )
            sma_window = c6.selectbox(
                "Forecast window (days)",
                model.ALLOWED_SMA_WINDOWS,
                index=model.ALLOWED_SMA_WINDOWS.index(assumptions.sma_window),
                help="How many recent days the moving-average forecast averages.",
            )

            b1, b2 = st.columns(2)
            recalc = b1.form_submit_button("Recalculate", width="stretch")
            reset = b2.form_submit_button("Reset to defaults", width="stretch")

        if recalc:
            try:
                st.session_state.assumptions = model.validate_assumptions(
                    ordering_cost=ordering_cost, holding_cost=holding_cost,
                    lead_time_days=lead_time, demand_change_pct=demand_change,
                    service_level=service_level, sma_window=sma_window,
                )
                st.rerun()
            except model.InputError as err:
                # Invalid input: explain and keep the previous values.
                st.error(f"Please fix the inputs: {err}")
        if reset:
            st.session_state.assumptions = model.DEFAULT_ASSUMPTIONS
            st.rerun()

    st.caption(
        "Data: H&M Personalized Fashion Recommendations (Kaggle), 18 steady-selling products, "
        f"real daily sales {sales_df.index.min():%Y-%m-%d} to {sales_df.index.max():%Y-%m-%d}. "
        "Current stock is simulated; costs and lead time are assumptions. "
        "Student project using public H&M data; not affiliated with or endorsed by H&M."
    )

# ==========================================================
# 6. RIGHT PANEL: RESTOCK ADVISOR (the AI chat)
# ==========================================================

with chat_col:

    st.markdown(
        '<div class="advisor-head">'
        '<h2 class="panel-title assist-title">Stock Pilot</h2>'
        '<p class="advisor-sub">Your AI inventory agent: ask what to reorder, how much and why.</p></div>',
        unsafe_allow_html=True)

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

    # Conversation, with the tools behind each answer shown for transparency.
    # The key gives the box a CSS class (st-key-advisor_chat) for its styling.
    chat_box = st.container(height=560, border=True, key="advisor_chat", autoscroll=True)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Show the photo of any product the advisor looked up.
                shown = {res.get("product_id") for name, _, res in msg.get("tool_calls", [])
                         if name == "get_product_details" and isinstance(res, dict)}
                for pid in sorted(p for p in shown if p):
                    if os.path.exists(model.image_path(pid)):
                        st.image(model.image_path(pid), width=110)
                if msg.get("tool_calls"):
                    with st.expander(f"Tools used ({len(msg['tool_calls'])})"):
                        for name, args, result in msg["tool_calls"]:
                            st.markdown(f"**{name}**(`{args}`)")
                            st.json(result, expanded=False)
                if msg.get("note"):
                    st.caption(msg["note"])

    # Suggested questions (useful for demos).
    suggestions = ["Which products should I prioritize?",
                   "What if demand increases by 20%?",
                   "Why does Lucy blouse need replenishment?"]
    picked = None
    st.markdown('<p class="try-label">Try asking</p>', unsafe_allow_html=True)
    for text in suggestions:   # stacked so the full question is always readable
        if st.button(text, width="stretch", key=f"suggest_{text}"):
            picked = text

    # A form keeps the input inside the right column.
    with st.form("chat_form", clear_on_submit=True, border=False):
        in_col, send_col = st.columns([3, 1], vertical_alignment="bottom")
        typed = in_col.text_input("Your question", placeholder="Ask Stock Pilot...",
                                  label_visibility="collapsed")
        sent = send_col.form_submit_button("Ask", width="stretch", type="primary")

    if st.button("Clear conversation", key="clear_chat", type="tertiary"):
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        st.rerun()

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
