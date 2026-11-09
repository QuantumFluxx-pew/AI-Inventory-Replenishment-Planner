"""AI Inventory Replenishment Planner | Milestone 2.

Run locally with: streamlit run app.py
Data file: inventory_replenishment_results.csv (same folder).
"""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="AI Inventory Replenishment Planner",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CLEAN, LIGHT VISUAL DESIGN ----------
st.markdown(
    """
    <style>
      .stApp { background: #f7f9fc; color: #17243a; }
      [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e6ebf2; }
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
      h1, h2, h3 { color: #17243a; letter-spacing: -.025em; }
      div[data-testid="stMetric"] {
          background: white; border: 1px solid #e6ebf2;
          border-radius: 14px; padding: 17px 20px;
          box-shadow: 0 3px 12px rgba(21, 42, 80, .035);
      }
      div[data-testid="stMetricLabel"] { color: #65748b; }
      div[data-testid="stMetricValue"] { color: #17243a; }
      div[data-testid="stChatMessage"] {
          background: white; border: 1px solid #e6ebf2;
          border-radius: 14px; padding: 12px 16px; margin-bottom: 10px;
      }
      .eyebrow { color: #3765b0; font-size: .83rem; font-weight: 700; letter-spacing: .1em; }
      .muted { color: #64748b; font-size: .95rem; }
      .section-title { font-size: 1.35rem; font-weight: 700; margin: 16px 0 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- LOAD AND VALIDATE THE INVENTORY CSV ----------
DATA_FILE = Path(__file__).resolve().parent / "inventory_replenishment_results.csv"
REQUIRED_COLUMNS = {
    "Product_ID", "SMA_7", "MAE", "Annual_Demand", "EOQ",
    "ROP", "Current_Stock", "Status", "Order_Quantity",
}


@st.cache_data
# Cache the CSV to avoid unnecessary disk reads on every interaction.
def load_inventory(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time  # Used only to refresh cache when the file changes.
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(sorted(missing)))

    frame["Product_ID"] = frame["Product_ID"].astype(str).str.strip()
    frame["Status"] = frame["Status"].astype(str).str.strip()
    for col in ["SMA_7", "MAE", "Annual_Demand", "EOQ", "ROP", "Current_Stock", "Order_Quantity"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return frame


if not DATA_FILE.exists():
    st.error("The inventory CSV was not found. Upload inventory_replenishment_results.csv to the same GitHub folder as app.py.")
    st.stop()

try:
    inventory = load_inventory(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except (ValueError, pd.errors.ParserError, OSError) as error:
    st.error(f"Could not load the inventory data: {error}")
    st.stop()

# ---------- SIDEBAR FILTERS ----------
with st.sidebar:
    st.markdown("### Inventory Control")
    st.caption("Milestone 2 | M5 sales forecasting")
    st.divider()
    status_filter = st.selectbox("Inventory status", ["All", "Reorder", "Sufficient Stock"])
    product_search = st.text_input("Search product ID", placeholder="e.g., HOBBIES_1_001")
    st.divider()
    st.caption("The results are based on the current CSV snapshot, not live warehouse inventory.")

filtered = inventory.copy()
if status_filter != "All":
    filtered = filtered[filtered["Status"].str.casefold() == status_filter.casefold()]
if product_search.strip():
    filtered = filtered[
        filtered["Product_ID"].str.contains(product_search.strip(), case=False, regex=False)
    ]

# ---------- PAGE HEADER ----------
st.markdown('<div class="eyebrow">INVENTORY INTELLIGENCE</div>', unsafe_allow_html=True)
st.title("AI Inventory Replenishment Planner")
st.markdown(
    '<div class="muted">Monitor stock, prioritize replenishment, and explore forecasting results.</div>',
    unsafe_allow_html=True,
)
st.write("")

# ---------- SUMMARY CARDS ----------
reorder_rows = filtered[filtered["Status"].str.casefold() == "reorder"]
sufficient_rows = filtered[filtered["Status"].str.casefold() == "sufficient stock"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Products shown", f"{len(filtered):,}")
c2.metric("Need reorder", f"{len(reorder_rows):,}")
c3.metric("Sufficient stock", f"{len(sufficient_rows):,}")
c4.metric("Recommended order units", f"{int(filtered['Order_Quantity'].sum()):,}")
st.caption("Summary cards reflect the current sidebar filters.")

# ---------- DASHBOARD CHARTS ----------
st.markdown('<div class="section-title">Inventory overview</div>', unsafe_allow_html=True)
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Stock status")
    counts = filtered["Status"].value_counts().reindex(["Reorder", "Sufficient Stock"], fill_value=0)
    st.bar_chart(counts.rename("Products"), color="#4b7bec")
with right:
    st.subheader("Largest recommended orders")
