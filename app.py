
"""AI Inventory Replenishment Planner | Milestone 2.

Run locally with: streamlit run app.py
Data file: inventory_replenishment_results.csv (same folder).
"""

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
      /* Force readable text when Streamlit Cloud is set to dark mode. */
      .stApp, [data-testid="stAppViewContainer"] {
          background: #f7f9fc !important; color: #17243a !important;
      }
      [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
          background: #ffffff !important;
          border-right: 1px solid #e6ebf2;
      }
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] small {
          color: #243047 !important;
      }
      /* Streamlit input widgets may retain a dark appearance. */
      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] [data-baseweb="input"] > div {
          background: #ffffff !important;
          color: #17243a !important;
          border-color: #cbd5e1 !important;
      }
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] [data-baseweb="select"] span {
          color: #17243a !important;
          -webkit-text-fill-color: #17243a !important;
      }
      [data-testid="stSidebar"] input::placeholder {
          color: #64748b !important;
          -webkit-text-fill-color: #64748b !important;
      }
      [data-testid="stAppViewContainer"] p,
      [data-testid="stAppViewContainer"] label {
          color: #334155;
      }
      .block-container {
          padding-top: 2rem;
          padding-bottom: 3rem;
          max-width: 1450px;
      }
      h1, h2, h3 {
          color: #17243a;
          letter-spacing: -.025em;
      }
      div[data-testid="stMetric"] {
          background: white;
          border: 1px solid #e6ebf2;
          border-radius: 14px;
          padding: 17px 20px;
          box-shadow: 0 3px 12px rgba(21, 42, 80, .035);
      }
      div[data-testid="stMetricLabel"] {
          color: #65748b;
      }
      div[data-testid="stMetricValue"] {
          color: #17243a;
      }
      div[data-testid="stChatMessage"] {
          background: white;
          border: 1px solid #e6ebf2;
          border-radius: 14px;
          padding: 12px 16px;
          margin-bottom: 10px;
      }
      .eyebrow {
          color: #3765b0;
          font-size: .83rem;
          font-weight: 700;
          letter-spacing: .1em;
      }
      .muted {
          color: #64748b;
          font-size: .95rem;
      }
      .section-title {
          font-size: 1.35rem;
          font-weight: 700;
          margin: 16px 0 10px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- LOAD AND VALIDATE THE INVENTORY CSV ----------
DATA_FILE = (
    Path(__file__).resolve().parent
    / "inventory_replenishment_results.csv"
)

REQUIRED_COLUMNS = {
    "Product_ID",
    "SMA_7",
    "MAE",
    "Annual_Demand",
    "EOQ",
    "ROP",
    "Current_Stock",
    "Status",
    "Order_Quantity",
}


# Cache the CSV to avoid unnecessary disk reads.
@st.cache_data
def load_inventory(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time  # Refreshes cache when the file changes.
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(sorted(missing))
        )

    frame["Product_ID"] = (
        frame["Product_ID"].astype(str).str.strip()
    )
    frame["Status"] = frame["Status"].astype(str).str.strip()

    numeric_columns = [
        "SMA_7",
        "MAE",
        "Annual_Demand",
        "EOQ",
        "ROP",
        "Current_Stock",
        "Order_Quantity",
    ]

    for col in numeric_columns:
        frame[col] = pd.to_numeric(
            frame[col], errors="coerce"
        ).fillna(0)

    return frame


if not DATA_FILE.exists():
    st.error(
        "The inventory CSV was not found. Upload "
        "inventory_replenishment_results.csv to the same "
        "GitHub folder as app.py."
    )
    st.stop()

try:
    inventory = load_inventory(
        str(DATA_FILE), DATA_FILE.stat().st_mtime
    )
except (ValueError, pd.errors.ParserError, OSError) as error:
    st.error(f"Could not load the inventory data: {error}")
    st.stop()

# ---------- SIDEBAR FILTERS ----------
with st.sidebar:
    st.markdown("### Inventory Control")
    st.caption("Milestone 2 | M5 sales forecasting")
    st.divider()

    status_filter = st.selectbox(
        "Inventory status",
        ["All", "Reorder", "Sufficient Stock"],
    )

    product_search = st.text_input(
        "Search product ID",
        placeholder="e.g., HOBBIES_1_001",
    )

    st.divider()
    st.caption(
        "The results are based on the current CSV snapshot, "
        "not live warehouse inventory."
    )

# Apply the selected filters.
filtered = inventory.copy()

if status_filter != "All":
    filtered = filtered[
        filtered["Status"].str.casefold()
        == status_filter.casefold()
    ]

if product_search.strip():
    filtered = filtered[
        filtered["Product_ID"].str.contains(
            product_search.strip(),
            case=False,
            regex=False,
        )
    ]

# ---------- PAGE HEADER ----------
st.markdown(
    '<div class="eyebrow">INVENTORY INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.title("AI Inventory Replenishment Planner")

st.markdown(
    '<div class="muted">Monitor stock, prioritize replenishment, '
    'and explore forecasting results.</div>',
    unsafe_allow_html=True,
)

st.write("")

# ---------- SUMMARY CARDS ----------
reorder_rows = filtered[
    filtered["Status"].str.casefold() == "reorder"
]

sufficient_rows = filtered[
    filtered["Status"].str.casefold() == "sufficient stock"
]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Products shown", f"{len(filtered):,}")
c2.metric("Need reorder", f"{len(reorder_rows):,}")
c3.metric("Sufficient stock", f"{len(sufficient_rows):,}")
c4.metric(
    "Recommended order units",
    f"{int(filtered['Order_Quantity'].sum()):,}",
)

st.caption("Summary cards reflect the current sidebar filters.")

# ---------- DASHBOARD CHARTS ----------
st.markdown(
    '<div class="section-title">Inventory overview</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Stock status")

    counts = (
        filtered["Status"]
        .value_counts()
        .reindex(
            ["Reorder", "Sufficient Stock"],
            fill_value=0,
        )
    )

    # Use a light chart theme.
    st.bar_chart(
        counts.rename("Products"),
        color="#4b7bec",
        theme=None,
    )

with right:
    st.subheader("Largest recommended orders")

    top_orders = reorder_rows.nlargest(
        8, "Order_Quantity"
    )

    if top_orders.empty:
        st.info(
            "No reorder recommendations match the current filters."
        )
    else:
        st.bar_chart(
            top_orders.set_index("Product_ID")["Order_Quantity"],
            color="#16a085",
            theme=None,
        )

# ---------- FILTERABLE INVENTORY TABLE ----------
st.markdown(
    '<div class="section-title">Product inventory</div>',
    unsafe_allow_html=True,
)

st.dataframe(
    filtered[
        [
            "Product_ID",
            "Current_Stock",
            "ROP",
            "EOQ",
            "Annual_Demand",
            "SMA_7",
            "MAE",
            "Status",
            "Order_Quantity",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Product_ID": "Product ID",
        "Current_Stock": st.column_config.NumberColumn(
            "Current stock", format="%d"
        ),
        "ROP": st.column_config.NumberColumn(
            "Reorder point", format="%d"
        ),
        "EOQ": st.column_config.NumberColumn(
            "EOQ", format="%d"
        ),
        "Annual_Demand": st.column_config.NumberColumn(
            "Annual demand", format="%.1f"
        ),
        "SMA_7": st.column_config.NumberColumn(
            "7-day average", format="%.2f"
        ),
        "MAE": st.column_config.NumberColumn(
            "Forecast MAE", format="%.2f"
        ),
        "Order_Quantity": st.column_config.NumberColumn(
            "Order units", format="%d"
        ),
    },
)

# Allow users to download the filtered results.
st.download_button(
    "Download filtered inventory (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_inventory.csv",
    mime="text/csv",
)

# ---------- DATA-BASED CHAT RESPONSE ENGINE ----------
def answer_question(question: str, data: pd.DataFrame) -> str:
    """Answer inventory questions using CSV data."""

    q = question.casefold().strip()

    if data.empty:
        return (
            "There are no products matching the current filters. "
            "Clear the filters and try again."
        )

    # Check whether the user mentioned a specific product ID.
    matches = [
        pid
        for pid in data["Product_ID"]
        if pid.casefold() in q
    ]

    if matches:
        pid = max(matches, key=len)
        row = data.loc[data["Product_ID"] == pid].iloc[0]

        return (
            f"**{pid}**\n\n"
            f"- Current stock: **{int(row['Current_Stock'])} units**\n"
            f"- Reorder point (ROP): **{int(row['ROP'])} units**\n"
            f"- Economic order quantity (EOQ): **{int(row['EOQ'])} units**\n"
            f"- Inventory status: **{row['Status']}**\n"
            f"- Recommended order: **{int(row['Order_Quantity'])} units**\n"
            f"- Estimated annual demand: **{row['Annual_Demand']:.1f} units**\n"
            f"- 7-day moving average: **{row['SMA_7']:.2f} units/day**\n"
            f"- Forecast MAE: **{row['MAE']:.2f} units/day**"
        )

    # Identify products requiring replenishment.
    reorder = data[
        data["Status"].str.casefold() == "reorder"
    ]

    sufficient = data[
        data["Status"].str.casefold() == "sufficient stock"
    ]

    # List the products available in the CSV.
    if any(
        term in q
        for term in [
            "what do we sell",
            "what all we sell",
            "which products",
            "list products",
            "product list",
            "all products",
        ]
    ):
        ids = ", ".join(
            data["Product_ID"].astype(str).tolist()
        )

        return (
            f"The dataset contains **{len(data)} product IDs**: "
            f"{ids}.\n\n"
            "The CSV does not contain product names or "
            "descriptions, so I cannot identify what these "
            "products actually are."
        )

    # Show which products need reordering.
    if (
        any(
            term in q
            for term in ["which", "list", "show", "top"]
        )
        and any(
            term in q
            for term in ["reorder", "restock", "order"]
        )
    ):
        if reorder.empty:
            return (
                "No products currently need reordering "
                "in the selected data."
            )

        items = "\n".join(
            f"- **{r.Product_ID}**: "
            f"stock {int(r.Current_Stock)}, "
            f"ROP {int(r.ROP)}, "
            f"order {int(r.Order_Quantity)} units"
            for r in reorder.itertuples(index=False)
        )

        return (
            f"**{len(reorder)} products need reordering:**"
            f"\n\n{items}"
        )

    # Count the products needing replenishment.
    if (
        any(
            term in q
            for term in ["how many", "number", "count"]
        )
        and any(
            term in q
            for term in ["reorder", "restock", "order"]
        )
    ):
        return (
            f"**{len(reorder)} products** need reordering, "
            f"with **{int(reorder['Order_Quantity'].sum()):,} "
            f"recommended units** in total."
        )

    # Calculate the total recommended order quantity.
    if any(
        term in q
        for term in [
            "total order",
            "order quantity",
            "how much to order",
            "units to order",
            "recommended order",
        ]
    ):
        return (
            "The total recommended order quantity is "
            f"**{int(data['Order_Quantity'].sum()):,} units** "
            "across the selected products."
        )

    # Summarize inventory stock levels.
    if any(
        term in q
        for term in [
            "stock level",
            "inventory level",
            "how much stock",
            "current stock",
            "stock summary",
        ]
    ):
        return (
            f"There are **{len(data)} products** in the "
            f"selected dataset. Current stock totals "
            f"**{int(data['Current_Stock'].sum()):,} units**. "
            f"**{len(reorder)}** need reordering and "
            f"**{len(sufficient)}** have sufficient stock."
        )

    # Summarize demand forecasting results.
    if any(
        term in q
        for term in [
            "forecast",
            "demand",
            "moving average",
            "mae",
        ]
    ):
        return (
            "The dataset includes 7-day moving average "
            "(SMA_7), forecast error (MAE), and estimated "
            f"annual demand for **{len(data)} products**. "
            f"Total estimated annual demand is "
            f"**{data['Annual_Demand'].sum():,.1f} units**. "
            "Ask about a specific product ID to see "
            "its forecast values."
        )

    # Count products with sufficient inventory.
    if any(
        term in q
        for term in [
            "sufficient",
            "healthy stock",
            "in stock",
        ]
    ):
        return (
            f"**{len(sufficient)} products** are marked "
            "as having sufficient stock in the selected dataset."
        )

    # Provide help for unsupported questions.
    return (
        "I can answer questions using the inventory CSV. "
        "Try asking:\n\n"
        "- How many products need reordering?\n"
        "- Which products need reordering?\n"
        "- What are the current stock levels?\n"
        "- What is the total recommended order quantity?\n"
        "- Tell me about HOBBIES_1_001\n\n"
        "This assistant uses dataset-based rules rather "
        "than a generative AI model."
    )


# ---------- CHAT PANEL ----------
st.divider()

st.markdown(
    '<div class="section-title">Ask the Inventory Assistant</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Ask about your products, forecasts, reorder points, "
    "and stock. Chat uses the currently filtered products."
)

# Store chat messages so they persist during interactions.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I can help you explore your inventory. "
                "What would you like to know?"
            ),
        }
    ]

# Display previous chat messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a new user question and generate a response.
if prompt := st.chat_input("Ask about your inventory..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    response = answer_question(prompt, filtered)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)

# Let the user reset the conversation.
if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

st.caption(
    "Prototype for Milestone 2. EOQ, ROP, and demand "
    "estimates are imported from the CSV; they are not "
    "recalculated in this app."
)
