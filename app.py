
# AI Inventory Replenishment Planner
# Milestone 2: Interactive Dashboard
# This dashboard displays demand forecasts, inventory levels,
# and replenishment recommendations for 20 selected products.

import streamlit as st
import pandas as pd

# 1. Set up the dashboard
# Use a wide layout so tables and charts are easier to read.
st.set_page_config(
    page_title="AI Inventory Replenishment Planner",
    page_icon="📦",
    layout="wide"
)

st.title("AI Inventory Replenishment Planner")
st.caption("Milestone 2 | M5 Sales Forecasting and Inventory Optimization")


# 2. Load the inventory results
# Read the CSV file generated from our Google Colab notebook.
# Cache the data so the file does not reload unnecessarily.
@st.cache_data
def load_data():
    return pd.read_csv("inventory_replenishment_results.csv")

df = load_data()


# 3. Add inventory status filters
# Allow users to view all products or filter by stock status.
st.sidebar.header("Filters")

status_options = ["All"] + sorted(df["Status"].unique().tolist())

selected_status = st.sidebar.selectbox(
    "Inventory Status",
    status_options
)

# Apply the selected filter to the dataset.
if selected_status == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[df["Status"] == selected_status].copy()


# 4. Calculate dashboard summary metrics
# Count products that need reordering or have sufficient stock.
total_products = len(filtered_df)
reorder_count = (filtered_df["Status"] == "Reorder").sum()
sufficient_count = (filtered_df["Status"] == "Sufficient Stock").sum()

# Calculate the total recommended ordering quantity.
total_order_units = int(filtered_df["Order_Quantity"].sum())

# Display the summary metrics in four columns.
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Products", total_products)
col2.metric("Need Reorder", reorder_count)
col3.metric("Sufficient Stock", sufficient_count)
col4.metric("Recommended Order Units", total_order_units)

st.divider()


# 5. Visualize inventory status
# Compare products needing reorder with products having enough stock.
st.subheader("Inventory Replenishment Status")

status_counts = filtered_df["Status"].value_counts()

st.bar_chart(status_counts)


# 6. Display reorder recommendations
# Show only products where current stock requires replenishment.
st.subheader("Reorder Recommendations")

reorder_df = filtered_df[
    filtered_df["Status"] == "Reorder"
]

# EOQ represents the recommended order size.
# ROP represents the inventory level that triggers a reorder.
st.dataframe(
    reorder_df[
        ["Product_ID", "Current_Stock", "ROP",
         "EOQ", "Order_Quantity"]
    ],
    use_container_width=True,
    hide_index=True
)


# 7. Display demand forecasting results
# SMA_7 is the 7-day Simple Moving Average forecast.
# MAE measures the average absolute forecasting error.
# Annual_Demand estimates yearly sales using daily demand.
st.subheader("Demand Forecasting Results")

st.dataframe(
    filtered_df[
        ["Product_ID", "SMA_7", "MAE",
         "Annual_Demand", "EOQ", "ROP"]
    ],
    use_container_width=True,
    hide_index=True
)


# 8. Visualize estimated daily demand
# Compare the 7-day SMA values across selected products.
st.subheader("Average Daily Demand by Product")

chart_data = filtered_df.set_index("Product_ID")["SMA_7"]

st.bar_chart(chart_data)


# 9. Allow users to download inventory recommendations
# Export the currently filtered results into a CSV file.
st.subheader("Download Results")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Inventory Recommendations",
    data=csv,
    file_name="inventory_recommendations.csv",
    mime="text/csv"
)


# 10. Explain the project assumptions
# The M5 dataset does not provide actual ordering costs,
# holding costs, or current inventory levels.
# Sample values are used to demonstrate the replenishment model.
st.info(
    "EOQ and reorder recommendations use sample cost "
    "and inventory assumptions for Milestone 2 testing."
)
