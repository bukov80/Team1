import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect("online_store.db")

query = """
SELECT
    o.order_id,
    o.customer_id,
    o.order_date,
    o.status,
    o.ship_country,
    c.segment,
    p.product_id,
    p.name AS product_name,
    cat.name AS category,
    oi.quantity,
    oi.unit_price,
    oi.discount,
    oi.quantity * oi.unit_price * (1 - oi.discount) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN order_items oi
    ON o.order_id = oi.order_id
JOIN products p
    ON oi.product_id = p.product_id
JOIN categories cat
    ON p.category_id = cat.category_id
"""

df = pd.read_sql(query, conn)


# Частка скасованих замовлень

orders_status = (
    df.groupby(["order_id", "status"])
    .size()
    .reset_index(name="cnt")
)

total_orders = orders_status["order_id"].nunique()

cancelled_orders = orders_status[
    orders_status["status"].str.lower() == "cancelled"
]["order_id"].nunique()

cancel_rate = cancelled_orders / total_orders * 100

print(f"Всього замовлень: {total_orders}")
print(f"Скасовано: {cancelled_orders}")
print(f"Частка скасованих: {cancel_rate:.2f}%")

# Категорія -> ймовірність скасування

category_cancel = (
    df.groupby(["category", "status"])
    .size()
    .reset_index(name="count")
)

category_pivot = (
    category_cancel
    .pivot(
        index="category",
        columns="status",
        values="count"
    )
    .fillna(0)
)

category_pivot["cancel_rate"] = (
    category_pivot["cancelled"]
    / category_pivot.sum(axis=1)
    * 100
)

print(category_pivot.sort_values(
    "cancel_rate",
    ascending=False
))

# Сегмент -> скасування

segment_cancel = (
    df.groupby(["segment", "status"])
    .size()
    .reset_index(name="count")
)

segment_pivot = (
    segment_cancel
    .pivot(
        index="segment",
        columns="status",
        values="count"
    )
    .fillna(0)
)

segment_pivot["cancel_rate"] = (
    segment_pivot["cancelled"]
    / segment_pivot.sum(axis=1)
    * 100
)

print(segment_pivot)

# Країна -> скасування

country_cancel = (
    df.groupby(["ship_country", "status"])
    .size()
    .reset_index(name="count")
)

country_pivot = (
    country_cancel
    .pivot(
        index="ship_country",
        columns="status",
        values="count"
    )
    .fillna(0)
)

country_pivot["cancel_rate"] = (
    country_pivot["cancelled"]
    / country_pivot.sum(axis=1)
    * 100
)

print(
    country_pivot.sort_values(
        "cancel_rate",
        ascending=False
    )
)

# Втрачений revenue

lost_revenue = (
    df[df["status"].str.lower() == "cancelled"]
    ["revenue"]
    .sum()
)

print("\nВтрачений revenue:")
print(round(lost_revenue, 2))

# Товари які скасовують найчастіше

most_cancelled_products = (
    df[df["status"].str.lower() == "cancelled"]
    .groupby("product_name")
    .agg(
        cancellations=("order_id", "count"),
        lost_revenue=("revenue", "sum")
    )
    .reset_index()
    .sort_values(
        "cancellations",
        ascending=False
    )
)

print("\nТоп товарів по скасуваннях:")
print(most_cancelled_products.head(20))

# Stacked Bar

category_status = (
    df.groupby(["category", "status"])
    .size()
    .unstack(fill_value=0)
)

category_status.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)

plt.title("Status по категоріях")
plt.xlabel("Category")
plt.ylabel("Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap скасувань по країнах

heatmap_data = (
    df.groupby(["ship_country", "status"])
    .size()
    .unstack(fill_value=0)
)

plt.figure(figsize=(10, 6))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".0f",
    cmap="Reds"
)

plt.title("Скасування по країнах")
plt.show()

conn.close()