import os
import ast
import time
import psutil
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from langchain_ollama import OllamaLLM

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Local Grocery AI Agent",
    layout="wide"
)

st.title("Local Grocery AI Agent")
st.caption("Local AI-powered decision support system for grocery retailers")

# ---------------------------------
# LOAD DATA
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sales_path = os.path.join(BASE_DIR, "Sales.csv")
inventory_path = os.path.join(BASE_DIR, "Inventory.csv")
expenses_path = os.path.join(BASE_DIR, "Expenses.csv")
knowledge_path = os.path.join(BASE_DIR, "Knowledge.txt")

@st.cache_data
def load_data():

    sales = pd.read_csv(sales_path)
    inventory = pd.read_csv(inventory_path)
    expenses = pd.read_csv(expenses_path)

    return sales, inventory, expenses

sales, inventory, expenses = load_data()

# ---------------------------------
# CLEAN DATA
# ---------------------------------

def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df

sales = clean_columns(sales)
inventory = clean_columns(inventory)
expenses = clean_columns(expenses)

sales = sales.rename(columns={
    "Item Description": "Product",
    "Units Sold": "Quantity",
    "Sales Amount": "Revenue",
    "Cost Amount": "Cost"
})

sales["Quantity"] = pd.to_numeric(sales["Quantity"], errors="coerce")

for col in ["Revenue", "Cost", "Profit"]:
    sales[col] = pd.to_numeric(
        sales[col].astype(str).str.replace(r"[^\d.-]", "", regex=True),
        errors="coerce"
    )

sales["Posting Date"] = pd.to_datetime(
    sales["Posting Date"],
    errors="coerce"
)

# ---------------------------------
# TOOL FUNCTIONS
# ---------------------------------

def get_top_products():
    return (
        sales.groupby("Product")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

def total_revenue():
    return sales["Revenue"].sum()

def total_profit():
    return sales["Profit"].sum()

def total_cost():
    return sales["Cost"].sum()

def total_expenses():

    amount_col = next(
        (col for col in expenses.columns if "amount" in col.lower()),
        None
    )

    if not amount_col:
        return "Amount column not found."

    return pd.to_numeric(
        expenses[amount_col],
        errors="coerce"
    ).sum()

def monthly_sales():
    monthly = sales.groupby(
        sales["Posting Date"].dt.to_period("M")
    )["Revenue"].sum()

    return monthly.sort_index()

def inventory_status():

    return (
        inventory.sort_values(
            by="Closing Stock",
            ascending=True
        )[["Item Description", "Closing Stock"]]
        .head(5)
    )

def slow_moving_products():

    return (
        inventory.sort_values(
            by="Qty Out",
            ascending=True
        )[["Item Description", "Qty Out"]]
        .head(5)
    )

def low_stock_alert(threshold=10):

    return inventory[
        inventory[inventory.columns[-1]] < threshold
    ]

# ---------------------------------
# LOAD LLM
# ---------------------------------

llm = OllamaLLM(model="mistral")

# ---------------------------------
# MEMORY
# ---------------------------------

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

def add_to_memory(question, answer):

    st.session_state.conversation_memory.append({
        "question": question,
        "answer": str(answer)
    })

def get_memory_context():

    if not st.session_state.conversation_memory:
        return "No previous context."

    history = ""

    for item in st.session_state.conversation_memory[-3:]:

        history += (
            f"Q: {item['question']}\n"
            f"A: {item['answer']}\n"
        )

    return history

# ---------------------------------
# SIMPLE RAG
# ---------------------------------

with open(knowledge_path, "r", encoding="utf-8") as f:
    knowledge = f.read()

def retrieve_context(question):

    if any(
        word in question.lower()
        for word in ["banana", "inventory", "stock", "perishable"]
    ):
        return knowledge

    return "No additional context available."

# ---------------------------------
# TOOL REGISTRY
# ---------------------------------

tools = {
    "top_products": get_top_products,
    "total_revenue": total_revenue,
    "total_profit": total_profit,
    "total_cost": total_cost,
    "total_expenses": total_expenses,
    "monthly_sales": monthly_sales,
    "inventory_status": inventory_status,
    "slow_moving_products": slow_moving_products,
    "low_stock_alert": low_stock_alert
}

tool_descriptions = {
    "top_products": "Best selling products",
    "total_revenue": "Total revenue",
    "total_profit": "Total profit",
    "total_cost": "Total costs",
    "total_expenses": "Operating expenses",
    "monthly_sales": "Monthly sales trends",
    "inventory_status": "Low inventory items",
    "slow_moving_products": "Products not selling",
    "low_stock_alert": "Critically low stock"
}

# ---------------------------------
# AGENT PLANNER
# ---------------------------------

def plan_steps(question):

    tool_list = "\n".join(
        [f"{k}: {v}" for k, v in tool_descriptions.items()]
    )

    prompt = f"""
    You are an AI planner.

    Tools:
    {tool_list}

    User Question:
    {question}

    Return ONLY a Python list.

    Examples:
    ["total_revenue"]
    ["monthly_sales"]
    ["inventory_status"]

    ONLY return the list.
    """

    response = llm.invoke(prompt)

    try:
        steps = ast.literal_eval(response)
        return [s for s in steps if s in tools]

    except:
        return []

# ---------------------------------
# EXECUTE TOOLS
# ---------------------------------

def execute_plan(steps):

    results = {}

    for step in steps:

        try:
            results[step] = tools[step]()

        except Exception as e:
            results[step] = f"Error: {e}"

    return results

# ---------------------------------
# USER INPUT
# ---------------------------------

user_question = st.text_input(
    "Ask your grocery business question:"
)

# ---------------------------------
# RUN AGENT
# ---------------------------------

if st.button("Run AI Agent"):

    if user_question:

        start_time = time.time()

        with st.spinner("Analyzing business data..."):

            steps = plan_steps(user_question)

            if not steps:
                st.error("I don't understand the question.")
                st.stop()

            results = execute_plan(steps)

            memory_context = get_memory_context()

            prompt = f"""
            You are a grocery store AI assistant.

            Conversation History:
            {memory_context}

            User Question:
            {user_question}

            Tool Results:
            {results}

            Tasks:
            - Answer clearly
            - Provide insights
            - Give recommendations
            """

            response = llm.invoke(prompt)

        end_time = time.time()

        latency = end_time - start_time
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        # -----------------------------
        # DISPLAY RESPONSE
        # -----------------------------

        st.subheader("AI Response")
        st.write(response)

        # -----------------------------
        # DISPLAY TOOL RESULTS
        # -----------------------------

        st.subheader("Tool Results")
        st.write(results)

        # -----------------------------
        # MONTHLY SALES CHART
        # -----------------------------

        if "monthly_sales" in steps:

            monthly_data = monthly_sales()

            fig, ax = plt.subplots()

            monthly_data.plot(ax=ax)

            ax.set_title("Monthly Revenue Trend")
            ax.set_xlabel("Month")
            ax.set_ylabel("Revenue")

            st.pyplot(fig)

        # -----------------------------
        # SYSTEM METRICS
        # -----------------------------

        st.subheader("System Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latency",
            f"{latency:.2f}s"
        )

        col2.metric(
            "CPU Usage",
            f"{cpu_usage}%"
        )

        col3.metric(
            "Memory Usage",
            f"{memory_usage}%"
        )

        # -----------------------------
        # SAVE METRICS
        # -----------------------------

        metrics = {
            "question": user_question,
            "latency_seconds": latency,
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory_usage
        }

        metrics_df = pd.DataFrame([metrics])

        metrics_df.to_csv(
            "system_metrics.csv",
            mode="a",
            header=not os.path.exists("system_metrics.csv"),
            index=False
        )

        add_to_memory(user_question, response)