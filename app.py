import streamlit as st
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Inventory Replenishment Planner",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CLEAN WHITE DASHBOARD STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f7f9fc;
        color: #202b43;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    [data-testid="stSidebar"] * {
        color: #202b43;
    }

    h1, h2, h3, p, label {
        color: #202b43;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #53627a;
    }

    [data-testid="stMetricValue"] {
        color: #202b43;
    }

    div[data-testid="stChatMessage"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
    }

    div[data-testid="stChatMessage"] p {
        color: #202b43;
    }

    .stButton button {
        border-radius: 9px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD INVENTORY DATA
# --------------------------------------------------

@st.cache_data
def load_inventory():
    # CSV must be stored alongside app.py in GitHub.
    data = pd.read_csv("inventory_replenishment_results.csv")

    # Remove accidental spaces from column names.
    data.columns = data.columns.str.strip()

    # Convert numerical fields into actual numbers.
    numeric_columns = [
        "SMA_7",
        "MAE",
        "Annual_Demand",
        "EOQ",
        "ROP",
        "Current_Stock",
        "Order_Quantity",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            ).fillna(0)

    data["Product_ID"] = data["Product_ID"].astype(str)
    data["Status"] = data["Status"].astype(str).str.strip()

    return data


try:
    df = load_inventory()
except Exception as error:
    st.error("Unable to load the inventory CSV file.")
    st.code(str(error))
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

with st.sidebar:
    st.title("Inventory Control")
    st.caption("M5 Sales Forecasting | Milestone 2")

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
        "Results are based on the uploaded CSV snapshot, "
        "not live warehouse inventory."
    )

# --------------------------------------------------
# FILTER INVENTORY DATA
# --------------------------------------------------

filtered_df = df.copy()

if status_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Status"] == status_filter
    ]

if product_search:
    filtered_df = filtered_df[
        filtered_df["Product_ID"].str.contains(
            product_search, case=False, regex=False
        )
    ]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("AI Inventory Replenishment Planner")

st.write(
    "Monitor inventory, prioritize replenishment, "
    "and ask questions about your products."
)

st.divider()

# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

total_products = len(filtered_df)

reorder_df = filtered_df[
    filtered_df["Status"] == "Reorder"
]

reorder_count = len(reorder_df)

sufficient_count = len(
    filtered_df[filtered_df["Status"] == "Sufficient Stock"]
)

recommended_units = int(
    reorder_df["Order_Quantity"].sum()
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Products Shown", total_products)

with col2:
    st.metric("Need Reorder", reorder_count)

with col3:
    st.metric("Sufficient Stock", sufficient_count)

with col4:
    st.metric(
        "Recommended Order Units",
        f"{recommended_units:,}",
    )

st.caption("Metrics reflect the selected inventory filters.")

# --------------------------------------------------
# INVENTORY CHARTS
# --------------------------------------------------

st.subheader("Inventory Overview")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Stock Status")

    if not filtered_df.empty:
        counts = (
            filtered_df["Status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Products")
        )

        # Use a simple DataFrame chart to avoid
        # the previous Streamlit chart TypeError.
        st.bar_chart(
            counts,
            x="Status",
            y="Products",
        )
    else:
        st.info("No products match the selected filters.")

with chart_col2:
    st.markdown("#### Largest Recommended Orders")

    if not reorder_df.empty:
        top_orders = (
            reorder_df.nlargest(10, "Order_Quantity")
            [["Product_ID", "Order_Quantity"]]
            .set_index("Product_ID")
        )

        st.bar_chart(top_orders)
    else:
        st.info("No reorder recommendations in this selection.")

# --------------------------------------------------
# INVENTORY TABLE
# --------------------------------------------------

st.subheader("Product Inventory Details")

display_columns = [
    "Product_ID",
    "Current_Stock",
    "ROP",
    "EOQ",
    "Annual_Demand",
    "Status",
    "Order_Quantity",
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True,
)

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

csv_output = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Inventory Results",
    data=csv_output,
    file_name="filtered_inventory_results.csv",
    mime="text/csv",
)

st.divider()

# --------------------------------------------------
# INVENTORY CHAT ASSISTANT
# --------------------------------------------------

st.subheader("AI Inventory Assistant")

st.caption(
    "Ask about product stock, reorder quantities, "
    "inventory status, or demand forecasts."
)


def inventory_assistant(question, data):
    """
    Answer common inventory questions using CSV data.

    This is a rule-based assistant, not a generative
    AI model. It does not need an external API key.
    """

    question_lower = question.lower().strip()

    if data.empty:
        return "There is no inventory data available."

    # Check whether the user mentioned a product ID.
    matching_products = [
        product
        for product in data["Product_ID"].tolist()
        if product.lower() in question_lower
    ]

    if matching_products:
        product_id = matching_products[0]

        product = data[
            data["Product_ID"] == product_id
        ].iloc[0]

        return (
            f"**Product:** {product_id}\n\n"
            f"- Current stock: {int(product['Current_Stock'])} units\n"
            f"- Reorder point: {int(product['ROP'])} units\n"
            f"- Economic order quantity: {int(product['EOQ'])} units\n"
            f"- Inventory status: {product['Status']}\n"
            f"- Recommended order: {int(product['Order_Quantity'])} units\n"
            f"- Annual demand estimate: {product['Annual_Demand']:.1f} units"
        )

    # Questions about items requiring replenishment.
    if any(word in question_lower for word in [
        "reorder", "replenish", "restock", "order"
    ]):
        products = data[data["Status"] == "Reorder"]

        if products.empty:
            return "No products currently require reordering."

        total = int(products["Order_Quantity"].sum())

        product_list = "\n".join(
            f"- {row['Product_ID']}: "
            f"{int(row['Order_Quantity'])} units"
            for _, row in products.iterrows()
        )

        return (
            f"**{len(products)} products require reordering.**\n\n"
            f"Total recommended order: **{total:,} units**.\n\n"
            f"{product_list}"
        )

    # Questions about stock levels.
    if any(word in question_lower for word in [
        "stock", "inventory", "available"
    ]):
        sufficient = len(
            data[data["Status"] == "Sufficient Stock"]
        )

        low_stock = len(
            data[data["Status"] == "Reorder"]
        )

        return (
            f"There are **{len(data)} products** in the dataset.\n\n"
            f"- Sufficient stock: {sufficient}\n"
            f"- Need reorder: {low_stock}\n"
            f"- Total units currently in stock: "
            f"{int(data['Current_Stock'].sum()):,}"
        )

    # Questions about demand forecasts.
    if any(word in question_lower for word in [
        "forecast", "demand", "sma", "prediction"
    ]):
        top = data.nlargest(5, "Annual_Demand")

        results = "\n".join(
            f"- {row['Product_ID']}: "
            f"{row['Annual_Demand']:.1f} estimated annual units"
            for _, row in top.iterrows()
        )

        return (
            "**Top five products by estimated annual demand:**\n\n"
            f"{results}\n\n"
            "These estimates come from the existing CSV."
        )

    # Questions about product catalog.
    if any(phrase in question_lower for phrase in [
        "what do we sell",
        "what all we sell",
        "list products",
        "all products",
        "product list",
    ]):
        product_list = "\n".join(
            f"- {product}"
            for product in data["Product_ID"].tolist()
        )

        return (
            "The dataset contains the following product IDs:\n\n"
            f"{product_list}\n\n"
            "Product names and descriptions are not available "
            "in this CSV."
        )

    return (
        "I can answer questions about your inventory data. "
        "Try asking:\n\n"
        "- Which products need reordering?\n"
        "- What is the stock level of HOBBIES_1_001?\n"
        "- Which products have the highest demand?\n"
        "- What products are in the dataset?"
    )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm your Inventory Assistant. "
                "Ask me about stock levels, reorder quantities, "
                "or demand forecasts."
            ),
        }
    ]

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input appears at the bottom of the page.
user_question = st.chat_input(
    "Ask about your inventory..."
)

if user_question:
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Answer using the full CSV rather than sidebar filters.
    answer = inventory_assistant(user_question, df)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
