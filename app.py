
import streamlit as st
import pandas as pd
import html

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Inventory Replenishment Planner",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# 2. CLEAN WHITE DESIGN
# ==========================================================

st.markdown("""
<style>
    /* Main application */
    .stApp {
        background: #f5f7fb;
        color: #202b43;
    }

    .block-container {
        max-width: 1800px;
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Keep dashboard text readable */
    h1, h2, h3, h4, p, label {
        color: #202b43;
    }

    /* Dashboard metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e8f0;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 3px 12px rgba(20, 35, 65, 0.03);
    }

    [data-testid="stMetricLabel"] {
        color: #66758b;
    }

    [data-testid="stMetricValue"] {
        color: #202b43;
    }

    /* White form controls */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #ffffff;
        color: #202b43;
        border: 1px solid #dbe3ef;
        border-radius: 9px;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input {
        color: #202b43;
    }

    /* Chat message cards */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e3e8f0;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stChatMessage"] p {
        color: #202b43;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 12px;
    }

    [data-testid="stChatInput"] textarea {
        color: #202b43;
    }

    /* General buttons */
    .stButton button,
    .stDownloadButton button {
        background: #ffffff;
        color: #244b91;
        border: 1px solid #dbe3ef;
        border-radius: 9px;
    }

    /* Small section description */
    .muted {
        color: #718096;
        font-size: 0.9rem;
    }

    /* Section separation */
    hr {
        border-color: #e3e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. LOAD AND VALIDATE INVENTORY DATA
# ==========================================================

@st.cache_data
def load_inventory():
    # The CSV file must be in the same GitHub folder as app.py.
    data = pd.read_csv("inventory_replenishment_results.csv")

    # Remove extra spaces in column headers.
    data.columns = data.columns.str.strip()

    required_columns = [
        "Product_ID",
        "SMA_7",
        "MAE",
        "Annual_Demand",
        "EOQ",
        "ROP",
        "Current_Stock",
        "Status",
        "Order_Quantity",
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing CSV columns: " + ", ".join(missing)
        )

    # Ensure numeric calculations work correctly.
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
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        ).fillna(0)

    data["Product_ID"] = (
        data["Product_ID"].astype(str).str.strip()
    )

    data["Status"] = (
        data["Status"].astype(str).str.strip()
    )

    return data


try:
    df = load_inventory()

except Exception as error:
    st.error("Unable to load inventory data.")
    st.code(str(error))
    st.stop()

# ==========================================================
# 4. RULE-BASED INVENTORY CHAT ASSISTANT
# ==========================================================

def inventory_assistant(question, data):
    """
    Answer inventory questions using the actual CSV.

    This assistant is rule-based. It does not call an AI API.
    """

    q = question.lower().strip()

    # Find a specific product ID mentioned in the question.
    product_matches = [
        pid for pid in data["Product_ID"]
        if pid.lower() in q
    ]

    if product_matches:
        pid = product_matches[0]

        item = data[
            data["Product_ID"] == pid
        ].iloc[0]

        return (
            f"**{pid}**\n\n"
            f"- Current stock: {item['Current_Stock']:,.0f} units\n"
            f"- Reorder point: {item['ROP']:,.0f} units\n"
            f"- EOQ: {item['EOQ']:,.0f} units\n"
            f"- Status: {item['Status']}\n"
            f"- Recommended order: "
            f"{item['Order_Quantity']:,.0f} units\n"
            f"- Annual demand: "
            f"{item['Annual_Demand']:,.1f} units"
        )

    # List products in the dataset.
    if any(phrase in q for phrase in [
        "what do we sell",
        "what all we sell",
        "list products",
        "all products",
        "product catalog",
        "product list",
    ]):
        product_list = "\n".join(
            f"- {pid}"
            for pid in data["Product_ID"]
        )

        return (
            f"**Products in the dataset:**\n\n"
            f"{product_list}\n\n"
            "The CSV contains product IDs, not product names."
        )

    # Show products with the highest demand.
    if any(word in q for word in [
        "forecast",
        "demand",
        "prediction",
        "highest selling",
        "highest demand",
    ]):
        top = data.nlargest(5, "Annual_Demand")

        lines = "\n".join(
            f"- {row['Product_ID']}: "
            f"{row['Annual_Demand']:,.1f} units/year"
            for _, row in top.iterrows()
        )

        return (
            "**Top 5 products by estimated annual demand:**\n\n"
            + lines
        )

    # Reorder questions.
    if any(word in q for word in [
        "reorder",
        "replenish",
        "restock",
        "order",
        "low stock",
    ]):
        items = data[data["Status"] == "Reorder"]

        if items.empty:
            return "No products are flagged for reordering."

        total = int(items["Order_Quantity"].sum())

        lines = "\n".join(
            f"- {row['Product_ID']}: "
            f"{row['Order_Quantity']:,.0f} units"
            for _, row in items.iterrows()
        )

        return (
            f"**{len(items)} products need reordering.**\n\n"
            f"Recommended total: **{total:,} units**.\n\n"
            f"{lines}"
        )

    # General stock questions.
    if any(word in q for word in [
        "stock",
        "inventory",
        "available",
        "units",
    ]):
        sufficient = (
            data["Status"] == "Sufficient Stock"
        ).sum()

        reorder = (
            data["Status"] == "Reorder"
        ).sum()

        total_units = int(
            data["Current_Stock"].sum()
        )

        return (
            f"**Inventory summary**\n\n"
            f"- Products: {len(data)}\n"
            f"- Need reorder: {reorder}\n"
            f"- Sufficient stock: {sufficient}\n"
            f"- Current units: {total_units:,}"
        )

    # Default guidance for unsupported questions.
    return (
        "I can help you analyze your inventory CSV.\n\n"
        "Try asking:\n\n"
        "- Which products need reordering?\n"
        "- What is the stock level of HOBBIES_1_001?\n"
        "- Which products have the highest demand?\n"
        "- List all products."
    )

# ==========================================================
# 5. APPLICATION HEADER
# ==========================================================

st.title("AI Inventory Replenishment Planner")

st.caption(
    "M5 Sales Forecasting | Inventory Optimization | Milestone 2"
)

st.divider()

# ==========================================================
# 6. TWO-PANEL LAYOUT
# ==========================================================

# Dashboard occupies approximately 65% of the page.
# Chat assistant occupies approximately 35%.
dashboard_col, chat_col = st.columns(
    [1.8, 1],
    gap="large"
)

# ==========================================================
# 7. LEFT PANEL: INVENTORY DASHBOARD
# ==========================================================

with dashboard_col:

    st.header("Inventory Dashboard")

    st.caption(
        "Track inventory performance and replenishment needs."
    )

    # Filters appear inside the dashboard rather than sidebar.
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        status_filter = st.selectbox(
            "Inventory status",
            ["All", "Reorder", "Sufficient Stock"]
        )

    with filter_col2:
        product_search = st.text_input(
            "Search product ID",
            placeholder="HOBBIES_1_001"
        )

    # Apply filters to dashboard only.
    filtered_df = df.copy()

    if status_filter != "All":
        filtered_df = filtered_df[
            filtered_df["Status"] == status_filter
        ]

    if product_search:
        filtered_df = filtered_df[
            filtered_df["Product_ID"].str.contains(
                product_search,
                case=False,
                regex=False
            )
        ]

    # Calculate dashboard KPIs.
    reorder_df = filtered_df[
        filtered_df["Status"] == "Reorder"
    ]

    sufficient_df = filtered_df[
        filtered_df["Status"] == "Sufficient Stock"
    ]

    total_products = len(filtered_df)
    reorder_count = len(reorder_df)
    sufficient_count = len(sufficient_df)

    recommended_units = int(
        reorder_df["Order_Quantity"].sum()
    )

    # Four metric cards in a 2x2 grid for readability.
    m1, m2 = st.columns(2)

    with m1:
        st.metric("Products Shown", total_products)

    with m2:
        st.metric("Need Reorder", reorder_count)

    m3, m4 = st.columns(2)

    with m3:
        st.metric("Sufficient Stock", sufficient_count)

    with m4:
        st.metric(
            "Recommended Order Units",
            f"{recommended_units:,}"
        )

    st.divider()

    # ------------------------------------------------------
    # INVENTORY CHARTS
    # ------------------------------------------------------

    st.subheader("Stock Status")

    if not filtered_df.empty:

        counts = (
            filtered_df["Status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Products")
        )

        # No custom chart color arguments:
        # this avoids the earlier Streamlit TypeError.
        st.bar_chart(
            counts,
            x="Status",
            y="Products"
        )

    else:
        st.info("No products match your filters.")

    st.subheader("Largest Recommended Orders")

    if not reorder_df.empty:

        top_orders = (
            reorder_df.nlargest(
                10, "Order_Quantity"
            )[["Product_ID", "Order_Quantity"]]
            .set_index("Product_ID")
        )

        st.bar_chart(top_orders)

    else:
        st.info("No reorder items in this selection.")

    # ------------------------------------------------------
    # PRODUCT INVENTORY TABLE
    # ------------------------------------------------------

    st.subheader("Product Inventory Details")

    table_columns = [
        "Product_ID",
        "Current_Stock",
        "ROP",
        "EOQ",
        "Annual_Demand",
        "Status",
        "Order_Quantity",
    ]

    # Streamlit renders the interactive table.
    st.dataframe(
        filtered_df[table_columns],
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    # Download filtered data.
    csv_bytes = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Inventory Results",
        data=csv_bytes,
        file_name="inventory_results_filtered.csv",
        mime="text/csv",
    )

    st.caption(
        "Inventory values are from the saved CSV snapshot."
    )

# ==========================================================
# 8. RIGHT PANEL: CHAT ASSISTANT
# ==========================================================

with chat_col:

    st.header("Inventory Assistant")

    st.caption(
        "Ask questions about stock, replenishment, "
        "and demand forecasts."
    )

    st.divider()

    # Initialize chat history once per user session.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I can help you explore your "
                    "inventory data. What would you like "
                    "to know?"
                ),
            }
        ]

    # Clear chat button.
    if st.button("Clear conversation"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Conversation cleared. "
                    "What would you like to know?"
                ),
            }
        ]

    # Show the conversation inside a dedicated chat panel.
    chat_container = st.container(
        height=570,
        border=True
    )

    with chat_container:
        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Using st.form keeps the input in the RIGHT column,
    # instead of Streamlit's default full-page chat footer.
    with st.form(
        "inventory_chat_form",
        clear_on_submit=True
    ):

        question = st.text_input(
            "Your question",
            placeholder="Ask about your inventory...",
        )

        submitted = st.form_submit_button(
            "Send message",
            use_container_width=True
        )

    if submitted and question.strip():

        # Add user question to history.
        st.session_state.messages.append({
            "role": "user",
            "content": question.strip()
        })

        # Generate answer from the full inventory dataset.
        answer = inventory_assistant(
            question,
            df
        )

        # Add assistant response to history.
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # Refresh chat display.
        st.rerun()
