# 1.7 Sales Performance співробітників
# Об'єднати employees, orders, order_items.
# Проаналізувати:
# Хто з менеджерів приносить найбільший revenue
# У кого найнижчий % скасувань
# Середній чек замовлень на співробітника
# Розподіл revenue по регіонах менеджерів
# Який менеджер залучив найбільше повторних клієнтів
# Візуалізація:
# Горизонтальний bar chart рейтингу
# Scatter plot (revenue vs cancellation %) по співробітниках


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "online_store.db"

conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    e.employee_id,
    e.first_name || ' ' || e.last_name AS employee_name,
    e.region,

    o.order_id,
    o.customer_id,
    o.status,

    oi.quantity,
    oi.unit_price,
    oi.discount

FROM employees e

JOIN orders o
    ON e.employee_id = o.employee_id

JOIN order_items oi
    ON o.order_id = oi.order_id
"""

df = pd.read_sql(query, conn)
conn.close()

# revenue
df["revenue"] = (
    df["quantity"] *
    df["unit_price"] *
    (1 - df["discount"])
)


# 1. Хто з менеджерів приносить найбільший revenue
employee_revenue = (
    df.groupby("employee_name", as_index=False)
      .agg(revenue=("revenue", "sum"))
      .sort_values("revenue", ascending=False)
)

print(employee_revenue)


# 2. У кого найнижчий % скасувань
orders_status = (
    df[["employee_name", "order_id", "status"]]
      .drop_duplicates()
)

cancellation = (
    orders_status.groupby("employee_name")
    .agg(
        total_orders=("order_id", "count"),
        cancelled_orders=("status",
                          lambda x: (x == "cancelled").sum())
    )
)

cancellation["cancellation_pct"] = (
    cancellation["cancelled_orders"]
    / cancellation["total_orders"]
    * 100
)

cancellation = (
    cancellation
    .sort_values("cancellation_pct")
)

print(cancellation)


# 3. Середній чек замовлень на співробітника
order_totals = (
    df.groupby(
        ["employee_name", "order_id"],
        as_index=False
    )
    .agg(order_value=("revenue", "sum"))
)

avg_check = (
    order_totals.groupby("employee_name")
    .agg(avg_check=("order_value", "mean"))
    .sort_values("avg_check", ascending=False)
)

print(avg_check)


# 4. Revenue по регіонах менеджерів
region_revenue = (
    df.groupby("region", as_index=False)
      .agg(revenue=("revenue", "sum"))
      .sort_values("revenue", ascending=False)
)

print(region_revenue)


# 5. Хто залучив найбільше повторних клієнтів
customer_orders = (
    df[["customer_id", "order_id"]]
      .drop_duplicates()
      .groupby("customer_id")
      .size()
      .reset_index(name="orders_count")
)

repeat_customers = customer_orders[
    customer_orders["orders_count"] > 1
]

repeat_df = (
    df[
        df["customer_id"].isin(
            repeat_customers["customer_id"]
        )
    ]
)

repeat_clients = (
    repeat_df.groupby("employee_name")
    .agg(
        repeat_customers=("customer_id",
                          "nunique")
    )
    .sort_values(
        "repeat_customers",
        ascending=False
    )
)

print(repeat_clients)


# 6. Горизонтальний рейтинг менеджерів
top_managers = employee_revenue.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_managers,
    x="revenue",
    y="employee_name"
)

plt.title("Managers Revenue Ranking")
plt.xlabel("Revenue")
plt.ylabel("Manager")

plt.tight_layout()
plt.show()


# 7. Scatter Plot (Revenue vs Cancellation %)
performance = (
    employee_revenue.merge(
        cancellation.reset_index(),
        on="employee_name"
    )
)

print(performance.head())

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=performance,
    x="cancellation_pct",
    y="revenue",
    s=120
)

plt.title(
    "Revenue vs Cancellation Rate"
)

plt.xlabel("Cancellation %")
plt.ylabel("Revenue")

plt.tight_layout()
plt.show()
