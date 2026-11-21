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
verify_numbers = agent_module.verify_numbers

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
    h2.dash-title { background: linear-gradient(90deg, #000000 0%, #3a3a3a 50%, #707070 100%);
                    -webkit-background-clip: text; background-clip: text;
                    -webkit-text-fill-color: transparent; color: transparent;
                    width: fit-content; }
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
    .st-key-advisor_chat { overflow-x: hidden !important; }
.settings-changed { font-size: 0.95rem; color: #4A4A47; margin: 0.25rem 0 0.75rem; }   /* one scroll: the chat box only */
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

    /* Number check under each answer */
    .verify { font-size: 0.78rem; padding: 3px 8px; margin: 4px 0 2px 0; display: inline-block;
              border: 1px solid; line-height: 1.4; }
    .verify-ok { color: #1d6b3a; background: #e7f6ec; border-color: #1d6b3a; }
    .verify-warn { color: #8a5a00; background: #fff4e0; border-color: #8a5a00; }

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

# Example upload files (also in data/example_upload/), offered for download.
def _read_example(name):
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "example_upload", name), "rb") as f:
            return f.read()
    except OSError:
        return b""


SAMPLE_SALES_CSV = _read_example("example_sales.csv")
SAMPLE_STOCK_CSV = _read_example("example_stock.csv")

# The user may replace the H&M sample with their own sales (Settings tab).
custom = st.session_state.get("custom_data")
if custom and custom.get("version") != model.MODEL_VERSION:
    custom = st.session_state.custom_data = None   # uploaded under an older app version
USING_SAMPLE = not custom
if custom:
    base_df, sales_df = custom["base"], custom["sales"]
    DATA_NOTE = (f"The user uploaded their own data ({custom['label']}): "
                 f"{len(base_df)} products, daily sales from {sales_df.index.min():%Y-%m-%d} "
                 f"to {sales_df.index.max():%Y-%m-%d}. " + custom["stock_note"])
else:
    DATA_NOTE = None
ID_LABEL = "H&M article ID" if USING_SAMPLE else "Product ID"
N_PRODUCTS = len(base_df)

# Current model inputs. Shared by the dashboard and the chatbot.
# Reset them if they come from an older version of the app.
if not isinstance(st.session_state.get("assumptions"), model.Assumptions):
    st.session_state.assumptions = model.DEFAULT_ASSUMPTIONS

if USING_SAMPLE:
    _who = ("for 18 H&M products, from basics like the *Brit Baby Tee* "
            "to the *Pluto RW slacks*")
else:
    _who = (f"for your {N_PRODUCTS} uploaded products, such as "
            f"*{base_df['Name'].iloc[0]}*")
GREETING = (
    f"Hello! I'm Stock Pilot, your AI inventory agent {_who}. I can tell "
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
            alt.Tooltip("Name", title="Product"), alt.Tooltip("Product_ID", title=ID_LABEL),
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
        tooltip=[alt.Tooltip("Name", title="Product"), alt.Tooltip("Product_ID", title=ID_LABEL),
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
    # Leave ~15% room past the longest bar so its value label is never cut off.
    x_max = float(top["Order_Quantity"].max() or 1) * 1.15
    base = alt.Chart(top).encode(
        y=alt.Y("Name:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)))
    bars = base.mark_bar(color=ORDER_BLUE, cornerRadiusEnd=4,
                         height={"band": 0.7}).encode(
        x=alt.X("Order_Quantity:Q", title="Units to order (EOQ)",
                scale=alt.Scale(domain=[0, x_max]), axis=alt.Axis(tickCount=6)),
        tooltip=[alt.Tooltip("Name", title="Product"), alt.Tooltip("Product_ID", title=ID_LABEL),
                 alt.Tooltip("Order_Quantity", title="Units to order")],
    )
    values = base.mark_text(align="left", dx=4, color="#202b43", fontSize=12).encode(
        x="Order_Quantity:Q", text=alt.Text("Order_Quantity:Q", format=","))
    return _style(bars + values, height=max(120, 28 * len(top)))


def sales_history_chart(daily_sales, window, alpha, uses_ses, days=90):
    """Actual daily units sold (bars) vs. both one-day-ahead forecasts (lines)
    for the last `days` days. The model in use is drawn solid and black; the
    other one dashed and grey, for comparison."""
    y = daily_sales.reset_index(drop=True)
    sma_name = f"{window}-day moving average"
    ses_name = f"Exponential smoothing (alpha {alpha:g})"
    frame = pd.DataFrame({"Date": daily_sales.index, "Actual units sold": daily_sales.values,
                          sma_name: model.sma_backtest(y, window).values,
                          ses_name: model.ses_backtest(y, alpha)}).tail(days)
    long = frame.melt("Date", var_name="Series", value_name="Units")
    used, other = (ses_name, sma_name) if uses_ses else (sma_name, ses_name)
    colors = alt.Scale(domain=["Actual units sold", used, other],
                       range=["#cfcfcb", "#111111", "#8a8984"])
    tips = [alt.Tooltip("Date:T"), alt.Tooltip("Series:N"), alt.Tooltip("Units:Q", format=".1f")]
    bars = alt.Chart(long[long["Series"] == "Actual units sold"]).mark_bar(width={"band": 0.8}).encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Units:Q", title="Units per day"),
        color=alt.Color("Series:N", scale=colors, legend=alt.Legend(orient="top", title=None, labelLimit=0)),
        tooltip=tips)
    other_line = alt.Chart(long[long["Series"] == other]).mark_line(
        strokeWidth=1.8, strokeDash=[5, 4]).encode(
        x="Date:T", y="Units:Q", color=alt.Color("Series:N", scale=colors), tooltip=tips)
    used_line = alt.Chart(long[long["Series"] == used]).mark_line(strokeWidth=2.5).encode(
        x="Date:T", y="Units:Q", color=alt.Color("Series:N", scale=colors), tooltip=tips)
    return _style(bars + other_line + used_line, height=260)

# ==========================================================
# 4. HEADER
# ==========================================================

st.markdown(
    '<div class="hero"><h1>AI Inventory Replenishment Planner</h1></div>',
    unsafe_allow_html=True,
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
    st.caption(f"Showing {summary['products']} of {len(plan)} products")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Need Reorder", summary["need_reorder"])
    k2.metric("Sufficient Stock", summary["sufficient_stock"])
    k3.metric("Units to Order", f"{summary['recommended_order_units']:,}")
    k4.metric("Annual Cost", f"${summary['total_annual_policy_cost']:,.0f}",
              help="Yearly ordering + holding cost of the EOQ policy for the products shown "
                   "(uses the assumed ordering and holding costs).")

    # A key keeps the open tab across reruns, so Recalculate stays on Settings.
    tab_overview, tab_products, tab_details, tab_settings = st.tabs(
        ["Overview", "Products", "Product Details", "Settings"], key="main_tabs", on_change="rerun")

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
                      "Order_Quantity", "EOQ", "Days_of_Cover", "Daily_Demand", "Forecast_Model", "MAE",
                      "Annual_Demand", "Annual_Cost", "Safety_Stock", "Product_ID"]
        st.dataframe(
            table[table_cols],
            width="stretch", hide_index=True, height=560, row_height=56,
            column_config={
                "Photo": st.column_config.ImageColumn("Photo", width="small"),
                "Name": st.column_config.TextColumn("Product"),
                "Product_Type": st.column_config.TextColumn("Type"),
                "Product_ID": st.column_config.TextColumn(ID_LABEL),
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
                "Forecast_Model": st.column_config.TextColumn("Forecast Model"),
                "Annual_Demand": st.column_config.NumberColumn("Annual Demand", format="%.0f"),
                "Annual_Cost": st.column_config.NumberColumn(
                    "Annual Cost", format="$%.0f",
                    help="Yearly ordering + holding cost when ordering EOQ units each time"),
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
        detail = InventoryTools(base_df, assumptions, DATA_NOTE).get_product_details(chosen)

        pic_col, info_col = st.columns([1, 3])
        with pic_col:
            if os.path.exists(model.image_path(chosen)):
                st.image(model.image_path(chosen), width="stretch")
        with info_col:
            st.markdown(" · ".join(f"**{detail['name']}**" if i == 0 else v for i, v in
                                   enumerate([detail["name"], detail["product_type"], detail["colour"]]) if v))
            if detail["description"]:
                st.caption(detail["description"])
            if USING_SAMPLE:
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
        cmp = detail["model_comparison_on_holdout"]
        uses_ses = detail["forecast_model"].startswith("Exponential")
        how = (f"exponential smoothing (alpha {cmp['exponential_smoothing_alpha']:g})"
               if uses_ses else f"average of the last {detail['sma_window_days']} days of sales")
        st.markdown(
            f"1. **Forecast:** {how} = "
            f"`{detail['sma_forecast_units_per_day']} units/day`\n"
            f"2. **Annual demand:** `{steps['annual_demand']}`\n"
            f"3. **EOQ:** `{steps['eoq']}`\n"
            f"4. **Reorder point:** `{steps['reorder_point']}`\n"
            f"5. **Decision:** `{steps['decision_rule']}`\n"
            f"6. **Annual cost:** `{steps['annual_cost']}`"
        )

        sma_hold = cmp[f"{detail['sma_window_days']}_day_moving_average_mae"]
        st.markdown(f"**Sales history vs. forecast (last 90 days)**: forecast accuracy "
                    f"(MAE over the year) = **{detail['forecast_mae_units_per_day']:.2f} units/day**")
        st.caption(
            f"Model comparison on the last 4 months (not used to fit either model): "
            f"moving average error {sma_hold:.2f}, exponential smoothing error "
            f"{cmp['exponential_smoothing_mae']:.2f} units/day, so "
            f"**{cmp['more_accurate_model']}** is more accurate for this product. "
            f"Forecast model in use: {detail['forecast_model']}.")
        chart = sales_history_chart(sales_df[chosen], assumptions.sma_window,
                                    cmp["exponential_smoothing_alpha"], uses_ses)
        st.altair_chart(chart, width="stretch", height=chart.height)

    # ------------------------------------------------------
    # 5e. SETTINGS TAB: model inputs (user can revise and rerun)
    # ------------------------------------------------------
    with tab_settings:
        st.caption(
            ("The H&M data has real sales but no costs, lead times or stock levels, "
             "so these are labelled assumptions. " if USING_SAMPLE else
             "Costs and lead time are assumptions you can change. ")
            + "Change them and click **Recalculate**; "
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

            c7, _, _ = st.columns(3)
            forecast_model = c7.selectbox(
                "Forecast model",
                model.ALLOWED_FORECAST_MODELS,
                index=model.ALLOWED_FORECAST_MODELS.index(assumptions.forecast_model),
                format_func=lambda k: model.FORECAST_MODEL_NAMES[k],
                help="Moving average, exponential smoothing, or the more accurate of the two "
                     "for each product (judged on the last 4 months of sales).",
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
                    forecast_model=forecast_model,
                )
                st.rerun()
            except model.InputError as err:
                # Invalid input: explain and keep the previous values.
                st.error(f"Please fix the inputs: {err}")
        if reset:
            st.session_state.assumptions = model.DEFAULT_ASSUMPTIONS
            st.rerun()

        # Results for the current settings, right under the form, so the effect
        # of Recalculate is visible without scrolling back up.
        st.markdown(f"**Results with these settings** (all {N_PRODUCTS} products)")
        all_summary = model.summarize(plan)
        modified = assumptions != model.DEFAULT_ASSUMPTIONS
        if modified:
            # Only the settings that differ from the defaults, as "old → new".
            d = model.DEFAULT_ASSUMPTIONS
            fields = [
                ("Ordering cost", d.ordering_cost, assumptions.ordering_cost, "${:g}"),
                ("Holding cost", d.holding_cost, assumptions.holding_cost, "${:g}"),
                ("Lead time", d.lead_time_days, assumptions.lead_time_days, "{:g} days"),
                ("Demand change", d.demand_change_pct, assumptions.demand_change_pct, "{:+g}%"),
                ("Safety stock", d.service_level, assumptions.service_level, "{:g}%"),
                ("Forecast window", d.sma_window, assumptions.sma_window, "{:g} days"),
            ]
            if d.forecast_model != assumptions.forecast_model:
                fields.append(("Forecast model", model.FORECAST_MODEL_NAMES[d.forecast_model],
                               model.FORECAST_MODEL_NAMES[assumptions.forecast_model], "{}"))
            changes = [f"{name} <b>{fmt.format(old)} → {fmt.format(new)}</b>"
                       for name, old, new, fmt in fields if old != new]
            st.markdown(
                '<p class="settings-changed">Changed from default: '
                + " &nbsp;·&nbsp; ".join(changes) + "</p>",
                unsafe_allow_html=True)
            base_summary = model.summarize(model.calculate_plan(base_df, d))

        def _delta(key):
            # Difference from the default results, shown under each number.
            if not modified:
                return None
            diff = all_summary[key] - base_summary[key]
            return f"{diff:+,} vs. default" if diff else "same as default"

        # Two rows of two, so the dollar amount is never cut off.
        r1, r2 = st.columns(2)
        r3, r4 = st.columns(2)
        r1.metric("Need Reorder", all_summary["need_reorder"],
                  delta=_delta("need_reorder"), delta_color="off")
        r2.metric("Sufficient Stock", all_summary["sufficient_stock"],
                  delta=_delta("sufficient_stock"), delta_color="off")
        r3.metric("Units to Order", f"{all_summary['recommended_order_units']:,}",
                  delta=_delta("recommended_order_units"), delta_color="off")
        cost_delta = None
        if modified:
            diff = all_summary["total_annual_policy_cost"] - base_summary["total_annual_policy_cost"]
            cost_delta = (f"{'+' if diff >= 0 else '-'}${abs(diff):,.0f} vs. default"
                          if round(diff) else "same as default")
        r4.metric("Annual Cost", f"${all_summary['total_annual_policy_cost']:,.0f}",
                  delta=cost_delta, delta_color="off",
                  help=f"Yearly ordering + holding cost of the EOQ policy for all {N_PRODUCTS} products.")
        if modified:
            st.caption("Click **Reset to defaults** to go back to the baseline.")
        to_order = plan[plan["Status"] == "Reorder"].sort_values("Days_of_Cover")
        if to_order.empty:
            st.success("No product needs reordering with these settings.")
        else:
            st.dataframe(
                to_order[["Name", "Days_of_Cover", "Current_Stock", "ROP", "Order_Quantity"]],
                hide_index=True, width="stretch",
                column_config={
                    "Name": "Product to reorder",
                    "Days_of_Cover": st.column_config.NumberColumn("Days of stock left", format="%.1f"),
                    "Current_Stock": "Stock",
                    "ROP": "Reorder point",
                    "Order_Quantity": st.column_config.NumberColumn("Order (EOQ)", format="localized"),
                },
            )

        # Upload your own data: same models, a store's own sales and stock.
        st.divider()
        st.markdown("**Use your own data**")
        if USING_SAMPLE:
            st.caption(
                "1. Download the templates and fill them in with your store's data "
                "(or use them as they are to try the feature).\n"
                "2. Upload them below and click **Use this data**.")
            f1, f2 = st.columns(2)
            with f1:
                st.markdown("**Sales template**: one row per product per day, at least 60 days")
                st.dataframe(pd.DataFrame({"date": ["2020-05-26", "2020-05-26", "2020-05-27"],
                                           "product": ["Brit Baby Tee", "Lucy blouse", "Brit Baby Tee"],
                                           "units": [28, 3, 20]}),
                             hide_index=True, width="stretch")
                st.download_button("Download sales template", SAMPLE_SALES_CSV,
                                   "sales_template.csv", "text/csv", width="stretch")
            with f2:
                st.markdown("**Stock template** (optional): units in stock now, one row per product")
                st.dataframe(pd.DataFrame({"product": ["Brit Baby Tee", "Lucy blouse"],
                                           "current_stock": [65, 180]}),
                             hide_index=True, width="stretch")
                st.download_button("Download stock template", SAMPLE_STOCK_CSV,
                                   "stock_template.csv", "text/csv", width="stretch")
            st.caption("Column names like item, sku, quantity, qty or stock also work, and extra "
                       "columns are ignored. Without a stock file, stock is estimated as 10 days "
                       "of recent sales.")
            with st.form("upload_form", border=False):
                u1, u2 = st.columns(2)
                sales_file = u1.file_uploader("Sales CSV", type="csv")
                stock_file = u2.file_uploader("Stock CSV (optional)", type="csv")
                use_upload = st.form_submit_button("Use this data", width="stretch")
            if use_upload:
                if sales_file is None:
                    st.error("Please choose a sales CSV first.")
                else:
                    try:
                        sales_raw = pd.read_csv(sales_file)
                        stock_raw = pd.read_csv(stock_file) if stock_file is not None else None
                        new_base, new_sales, notes = model.prepare_uploaded_data(sales_raw, stock_raw)
                    except model.InputError as err:
                        st.error(str(err))
                    except Exception as err:   # unreadable file, wrong encoding, ...
                        st.error(f"Could not read the file: {err}")
                    else:
                        st.session_state.custom_data = {
                            "base": new_base, "sales": new_sales, "notes": notes,
                            "label": sales_file.name, "version": model.MODEL_VERSION,
                            "stock_note": ("Current stock comes from the user's stock file."
                                           if stock_file is not None else
                                           "No stock file: stock is estimated as 10 days of "
                                           "recent sales, not real stock."),
                        }
                        st.session_state.assumptions = model.DEFAULT_ASSUMPTIONS
                        st.session_state.pop("messages", None)   # new products, new chat
                        st.rerun()
        else:
            st.success(f"Using your data: **{custom['label']}**, {N_PRODUCTS} products, "
                       f"{len(sales_df)} days.")
            for note in custom["notes"]:
                st.caption(note)
            if st.button("Back to the H&M sample data", width="stretch"):
                st.session_state.custom_data = None
                st.session_state.assumptions = model.DEFAULT_ASSUMPTIONS
                st.session_state.pop("messages", None)
                st.rerun()

    if USING_SAMPLE:
        st.caption(
            f"Data: H&M sales (Kaggle), {sales_df.index.min():%b %Y} to {sales_df.index.max():%b %Y} · "
            "Stock is simulated · Not affiliated with H&M"
        )
    else:
        st.caption(f"Data: your upload ({custom['label']}), {sales_df.index.min():%b %d %Y} to "
                   f"{sales_df.index.max():%b %d %Y} · {N_PRODUCTS} products")

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
                # Escape "$" so money amounts never turn into math formatting.
                st.markdown(msg["content"].replace("$", "\\$"))
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
                # Proof that the AI did not invent numbers (see agent.verify_numbers).
                check = msg.get("check")
                if check:
                    n = check["checked"]
                    if not check["unmatched"]:
                        label = (f"✓ Numbers checked: all {n} come from the model's calculations"
                                 if n else "✓ No calculated numbers in this answer")
                        if check.get("examples"):
                            label += f" ({', '.join(check['examples'])} were suggested examples)"
                        st.markdown(f'<span class="verify verify-ok">{label}</span>',
                                    unsafe_allow_html=True)
                    else:
                        bad = ", ".join(check["unmatched"])
                        st.markdown(
                            f'<span class="verify verify-warn">Check: {len(check["unmatched"])} of {n} '
                            f"numbers were not found in the model's calculations ({bad})</span>",
                            unsafe_allow_html=True)
                if msg.get("note"):
                    st.caption(msg["note"])

    # Suggested questions (useful for demos).
    example = "Lucy blouse" if USING_SAMPLE else model.prioritize(plan)["Name"].head(1).tolist()
    example = example if isinstance(example, str) else (example[0] if example else base_df["Name"].iloc[0])
    suggestions = ["Which products should I prioritize?",
                   "What if demand increases by 20%?",
                   f"Why does {example} need replenishment?"]
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

        tools = InventoryTools(base_df, st.session_state.assumptions, DATA_NOTE)
        note = None

        with st.spinner("Thinking..."):
            if AI_ENABLED:
                try:
                    reply = GeminiAgent(API_KEY, MODEL_NAME, data_note=DATA_NOTE).ask(question, history, tools)
                except Exception as err:
                    # Never fail silently: say Gemini failed and answer offline.
                    reply = OfflineAssistant().ask(question, history, tools)
                    note = (f"Gemini request failed ({type(err).__name__}: {str(err)[:150]}). "
                            "This answer came from offline mode.")
            else:
                reply = OfflineAssistant().ask(question, history, tools)

        # If the agent changed the assumptions, the dashboard follows.
        st.session_state.assumptions = tools.assumptions

        # Check every number in the answer against the model's results
        # (this answer's tool calls plus earlier ones in the conversation).
        earlier = [c for m in st.session_state.messages for c in m.get("tool_calls", [])]
        # (Offline help text with no tool calls only lists example questions.)
        check = (verify_numbers(reply.text, reply.tool_calls, question, earlier,
                                product_names=list(base_df["Name"]))
                 if reply.tool_calls or reply.mode == "llm" else None)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply.text,
            "tool_calls": reply.tool_calls,
            "check": check,
            "note": note,
        })
        st.rerun()
