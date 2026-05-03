import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain_ollama import OllamaLLM
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Grocery AI Dashboard", layout="wide")

st.title("🛒 Grocery AI Assistant")

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_data():
    sales = pd.read_csv("Sales.csv")
    inventory = pd.read_csv("Inventory.csv")
    expenses = pd.read_csv("Expenses.csv")

    def clean(df):
        df.columns = df.columns.str.strip()
        return df

    return clean(sales), clean(inventory), clean(expenses)

sales, inventory, expenses = load_data()

# -----------------------
# CLEANING
# -----------------------
sales = sales.rename(columns={
    "Item Description": "Product",
    "Units Sold": "Quantity",
    "Sales Amount": "Revenue",
    "Cost Amount": "Cost"
})

sales["Quantity"] = pd.to_numeric(sales["Quantity"], errors="coerce")
sales["Revenue"] = pd.to_numeric(sales["Revenue"], errors="coerce")
sales["Cost"] = pd.to_numeric(sales["Cost"], errors="coerce")
sales["Posting Date"] = pd.to_datetime(sales["Posting Date"], errors="coerce")

# -----------------------
# SIDEBAR FILTERS
# -----------------------
st.sidebar.header("🔎 Filters")

products = st.sidebar.multiselect(
    "Select Product",
    options=sales["Product"].dropna().unique()
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [sales["Posting Date"].min(), sales["Posting Date"].max()]
)

# Apply filters
filtered = sales.copy()

if products:
    filtered = filtered[filtered["Product"].isin(products)]

if len(date_range) == 2:
    filtered = filtered[
        (filtered["Posting Date"] >= pd.to_datetime(date_range[0])) &
        (filtered["Posting Date"] <= pd.to_datetime(date_range[1]))
    ]

# -----------------------
# KPI CARDS
# -----------------------
total_revenue = filtered["Revenue"].sum()
total_cost = filtered["Cost"].sum()
profit = total_revenue - total_cost

col1, col2, col3 = st.columns(3)

col1.metric("💰 Revenue", f"${total_revenue:,.0f}")
col2.metric("📉 Cost", f"${total_cost:,.0f}")
col3.metric("📈 Profit", f"${profit:,.0f}")

# -----------------------
# INTERACTIVE CHART
# -----------------------
monthly = (
    filtered.groupby(filtered["Posting Date"].dt.to_period("M"))["Revenue"]
    .sum()
    .reset_index()
)
monthly["Posting Date"] = monthly["Posting Date"].astype(str)

fig = px.line(monthly, x="Posting Date", y="Revenue", title="Monthly Revenue Trend")
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# AUTO INSIGHTS
# -----------------------
st.subheader("Auto Insights")

if not monthly.empty:
    best_month = monthly.loc[monthly["Revenue"].idxmax()]

    st.info(f"""
    - 📊 Best month: **{best_month['Posting Date']}**
    - 💰 Revenue: **${best_month['Revenue']:,.0f}**
    - 📈 Trend: {'Growing' if monthly['Revenue'].iloc[-1] > monthly['Revenue'].iloc[0] else 'Declining'}
    """)

# -----------------------
# SMART ALERTS
# -----------------------
st.subheader("Alerts")

low_stock = inventory[inventory["Closing Stock"] < 10]

if not low_stock.empty:
    st.warning(f"{len(low_stock)} products are low in stock!")
    st.dataframe(low_stock[["Item Description", "Closing Stock"]].head(5))
else:
    st.success("All inventory levels are healthy")

# -----------------------
# TOP PRODUCTS
# -----------------------
# st.subheader("🏆 Top Products")

# top_products = (
#    filtered.groupby("Product")["Quantity"]
#    .sum()
#    .sort_values(ascending=False)
#    .head(10)
# )

# st.dataframe(top_products)

# -----------------------
# DOWNLOAD CSV
# -----------------------
# st.download_button(
#    "⬇️ Download Sales",
#    filtered.to_csv(index=False),
#    file_name="sales.csv"
# )

# -----------------------
# DOWNLOAD PDF
# -----------------------
def create_pdf():
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Revenue: ${total_revenue:,.0f}", styles["Normal"]))
    content.append(Paragraph(f"Profit: ${profit:,.0f}", styles["Normal"]))

    doc.build(content)

    with open("report.pdf", "rb") as f:
        return f.read()

st.download_button(
    "⬇️ Download Report",
    create_pdf(),
    file_name="report.pdf"
)

# -----------------------
# AI ASSISTANT
# -----------------------
# st.subheader("Ask Your AI Assistant")

# -----------------------
# HYBRID LLM SETUP
# -----------------------

USE_CLOUD_LLM = os.getenv("OPENAI_API_KEY") is not None

if USE_CLOUD_LLM:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
else:
    from langchain_ollama import OllamaLLM

    llm = OllamaLLM(model="mistral")


# -----------------------
# AI ASSISTANT (SAFE)
# -----------------------
st.subheader("Ask Your AI Assistant")

user_q = st.text_input("Ask a business question:")

if st.button("Ask AI"):

    if not user_q:
        st.warning("Please enter a question.")
    else:
        try:
            response = llm.invoke(f"""
            You are a grocery business analyst.

            Data summary:
            Revenue = {total_revenue}
            Profit = {profit}

            Question:
            {user_q}

            Provide:
            - Clear answer
            - Business insight
            - 2 actionable recommendations
            """)

            st.success("AI Response")
            st.write(response)

        except Exception as e:
            st.error("AI is not available in this environment.")
            st.info("Run locally to enable full AI (Ollama).")
