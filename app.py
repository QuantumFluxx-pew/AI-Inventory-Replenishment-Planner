
# --------------------------------------------------
# AI Inventory Assistant
# Allows users to ask questions about inventory
# --------------------------------------------------

st.divider()
st.header("AI Inventory Assistant")

# Store previous messages during the session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept a question from the user
question = st.chat_input("Ask about your inventory...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # Initial rule-based responses using inventory data
    with st.chat_message("assistant"):
        q = question.lower()

        if "reorder" in q or "restock" in q:
            reorder_items = df[df["Status"] == "Reorder"]
            response = (
                f"{len(reorder_items)} products need reordering. "
                f"Recommended total: "
                f"{int(reorder_items['Order_Quantity'].sum())} units."
            )

        elif "stock" in q:
            response = (
                f"{len(df)} products are currently displayed. "
                f"{(df['Status'] == 'Sufficient Stock').sum()} "
                f"have sufficient stock."
            )

        else:
            response = (
                "I can help you check inventory status, "
                "reorder recommendations, and stock levels. "
                "Try asking: Which products need reordering?"
            )

        st.write(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
