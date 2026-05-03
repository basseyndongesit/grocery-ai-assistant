import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
from langchain_ollama import OllamaLLM

print("RUNNING FILE:", os.path.abspath(__file__))
print("PROGRAM STARTED")


# LOAD DATA

sales = pd.read_csv(r"C:\Users\basse\OneDrive\Documents\ANALYTICS spring 2025\SPS Project - Local AI Agent\Grocery AI Project\Sales.csv")
inventory = pd.read_csv(r"C:\Users\basse\OneDrive\Documents\ANALYTICS spring 2025\SPS Project - Local AI Agent\Grocery AI Project\Inventory.csv")
expenses = pd.read_csv(r"C:\Users\basse\OneDrive\Documents\ANALYTICS spring 2025\SPS Project - Local AI Agent\Grocery AI Project\Expenses.csv")

# After loading ALL datasets
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

sales.columns = sales.columns.str.strip()
inventory.columns = inventory.columns.str.strip()
expenses.columns = expenses.columns.str.strip()

# print("EXPENSE COLUMNS RAW:", list(expenses.columns))
# print("EXPENSE COLUMNS STRIPPED:", list(expenses.columns.str.strip()))

# CLEAN DATA

sales = sales.rename(columns={
    "Item Description": "Product",
    "Units Sold": "Quantity"
})

sales["Quantity"] = pd.to_numeric(sales["Quantity"], errors="coerce")

# Clean money columns
for col in ["Sales Amount", "Cost Amount", "Profit"]:
    sales[col] = pd.to_numeric(
        sales[col].astype(str).str.replace(r"[^\d.-]", "", regex=True),
        errors="coerce"
    )

# Rename
sales = sales.rename(columns={
    "Sales Amount": "Revenue",
    "Cost Amount": "Cost"
})

sales["Posting Date"] = pd.to_datetime(sales["Posting Date"], errors="coerce")

# DEFINE TOOLS

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

def total_expenses():
    expenses.columns = expenses.columns.str.strip()

    # dynamically find Amount column
    amount_col = next(
        (col for col in expenses.columns if "amount" in col.lower()),
        None
    )

    if not amount_col:
        return f"Error: Amount column not found. Columns: {list(expenses.columns)}"

    return pd.to_numeric(expenses[amount_col], errors="coerce").sum()

def total_cost():
    return sales["Cost"].sum()

def monthly_sales():
    monthly = sales.groupby(sales["Posting Date"].dt.to_period("M"))["Revenue"].sum()
    return monthly.sort_index()

def inventory_status():
    inventory.columns = inventory.columns.str.strip()

    return (
        inventory.sort_values(by="Closing Stock", ascending=True)
        [["Item Description", "Closing Stock"]]
        .head(5)
    )

def slow_moving_products():
    inventory.columns = inventory.columns.str.strip()

    return (
        inventory.sort_values(by="Qty Out", ascending=True)
        [["Item Description", "Qty Out"]]
        .head(5)
    )

# VISUAL TOOL (AUTO CHART)

def plot_monthly_sales():
    data = monthly_sales()

    plt.figure()
    data.plot()
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return "Chart displayed"

# SMART ALERTS

def low_stock_alert(threshold=10):
    low_stock = inventory[inventory[inventory.columns[-1]] < threshold]
    return low_stock

# LOAD AI MODEL

llm = OllamaLLM(model="mistral")

# MEMORY (CONVERSATION)

conversation_memory = []

def add_to_memory(question, answer):
    conversation_memory.append({
        "question": question,
        "answer": str(answer)
    })

def get_memory_context():
    if not conversation_memory:
        return "No previous context."
    
    history = ""
    for item in conversation_memory[-3:]:
        history += f"Q: {item['question']}\nA: {item['answer']}\n"
    
    return history

# SIMPLE RAG (KNOWLEDGE FILE)

with open("Knowledge.txt") as f:
    knowledge = f.read()

def retrieve_context(question):
    if any(word in question.lower() for word in ["banana", "inventory", "stock", "perishable"]):
        return knowledge
    return "No additional business context available."


# TOOL REGISTRY (AGENT)

tools = {
    "top_products": get_top_products,
    "total_revenue": total_revenue,
    "total_profit": total_profit,
    "total_cost": total_cost,
    "total_expenses": total_expenses,
    "monthly_sales": monthly_sales,
    "plot_monthly_sales": plot_monthly_sales,
    "inventory_status": inventory_status,
    "low_stock_alert": low_stock_alert
}

tool_descriptions = {
    "top_products": "Best selling products",
    "total_revenue": "Total revenue",
    "total_profit": "Total profit",
    "total_cost": "Total cost of goods",
    "total_expenses": "Operating expenses",
    "monthly_sales": "Monthly revenue data",
    "plot_monthly_sales": "Plot revenue trend chart",
    "inventory_status": "Low stock items",
    "low_stock_alert": "Detect critically low stock"
}

# TOOL SELECTION LOGIC

import re

def choose_tool_llm(question):
    tool_list = "\n".join([f"{k}: {v}" for k, v in tool_descriptions.items()])

    prompt = f"""
    You are an AI assistant that selects the best tool.

    Available tools:
    {tool_list}

    User question:
    {question}

    Rules:
    - Return ONLY the tool name
    - Do NOT explain
    - If no tool applies, return: NONE
    """

    response = llm.invoke(prompt).strip().lower()

    print("LLM selected tool:", response) 

    return response if response in tools else None

#  AGENT BRAIN (PLANNING + EXECUTION)

def plan_steps(question):
    tool_list = "\n".join([f"{k}: {v}" for k, v in tool_descriptions.items()])

    prompt = f"""
    You are an AI planner.

    Tools:
    {tool_list}

    User Question:
    {question}

    Return a Python list of tools to use.

    Examples:
    ["monthly_sales", "plot_monthly_sales"]
    ["inventory_status", "low_stock_alert"]
    ["total_revenue"]

    ONLY return list.
    """

    response = llm.invoke(prompt)

    try:
        steps = ast.literal_eval(response)
        return [s for s in steps if s in tools]
    except:
        return []

def execute_plan(steps):
    results = {}

    for step in steps:
        try:
            results[step] = tools[step]()
        except Exception as e:
            results[step] = f"Error: {e}"

    return results

# MAIN SYSTEM (AGENT FLOW)
print("REACHED INPUT")
print("PROGRAM STARTED")

while True:
    user_question = input("\nAsk a question (or type 'exit'): ")

    if user_question.lower() == "exit":
        print("Goodbye!")
        break

    steps = plan_steps(user_question)

#    print("PLANNED STEPS:", steps)  # 🔍 debug

    if not steps:
        print("I don't understand the question yet.")
        continue

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

    print("\nAI RESPONSE:\n")
    print(response)

    add_to_memory(user_question, response)