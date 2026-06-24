import sqlite3
import pandas as pd
import numpy as np
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
    p.name AS product_name,
    cat.name AS category,
    oi.quantity,
    oi.unit_price,
    oi.discount,
    oi.quantity * oi.unit_price * (1 - oi.discount) AS item_revenue
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id
JOIN products p
    ON oi.product_id = p.product_id
JOIN categories cat
    ON p.category_id = cat.category_id
JOIN customers c
    ON o.customer_id = c.customer_id
"""

df = pd.read_sql(query, conn)

# 1. Вартість кожного замовлення

orders_value = (
    df.groupby(["order_id", "customer_id", "order_date", "status", "ship_country", "segment"])
    .agg(
        order_value=("item_revenue", "sum"),
        items_count=("quantity", "sum"),
        categories_count=("category", "nunique")
    )
    .reset_index()
)

print("Перші рядки вартості замовлень:")
print(orders_value.head())

# 2. IQR: пошук викидів

q1 = orders_value["order_value"].quantile(0.25)
q2 = orders_value["order_value"].quantile(0.50)
q3 = orders_value["order_value"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

orders_value["outlier_iqr"] = (
    (orders_value["order_value"] < lower_bound) |
    (orders_value["order_value"] > upper_bound)
)

iqr_outliers = orders_value[orders_value["outlier_iqr"]]

print("\nIQR межі:")
print("Q1:", q1)
print("Median:", q2)
print("Q3:", q3)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

print("\nКількість IQR-викидів:")
print(len(iqr_outliers))

print("\nТоп IQR-викидів:")
print(
    iqr_outliers
    .sort_values("order_value", ascending=False)
    .head(10)
)

# 3. Z-score: пошук викидів

mean_value = orders_value["order_value"].mean()
std_value = orders_value["order_value"].std()

orders_value["z_score"] = (
    (orders_value["order_value"] - mean_value) / std_value
)

orders_value["outlier_z"] = orders_value["z_score"].abs() > 3

z_outliers = orders_value[orders_value["outlier_z"]]

print("\nКількість Z-score викидів:")
print(len(z_outliers))

print("\nТоп Z-score викидів:")
print(
    z_outliers
    .sort_values("z_score", ascending=False)
    .head(10)
)

# 4. Гістограма з квартилями

plt.figure(figsize=(12, 6))

sns.histplot(
    data=orders_value,
    x="order_value",
    bins=50,
    kde=True
)

plt.axvline(q1, linestyle="--", label="Q1")
plt.axvline(q2, linestyle="-", label="Median")
plt.axvline(q3, linestyle="--", label="Q3")
plt.axvline(upper_bound, linestyle=":", label="IQR upper bound")

plt.title("Розподіл вартості замовлень")
plt.xlabel("Order Value")
plt.ylabel("Кількість замовлень")
plt.legend()
plt.show()

# 5. Box plot по сегментах

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=orders_value,
    x="segment",
    y="order_value"
)

plt.title("Box plot вартості замовлень по сегментах")
plt.xlabel("Segment")
plt.ylabel("Order Value")
plt.xticks(rotation=45)
plt.show()

# 6. Box plot по країнах

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=orders_value,
    x="ship_country",
    y="order_value"
)

plt.title("Box plot вартості замовлень по країнах")
plt.xlabel("Country")
plt.ylabel("Order Value")
plt.xticks(rotation=45)
plt.show()

# 7. Топ-1% найбільших замовлень

top_1_percent_limit = orders_value["order_value"].quantile(0.99)

top_orders = orders_value[
    orders_value["order_value"] >= top_1_percent_limit
]

print("\nМежа топ-1% замовлень:")
print(top_1_percent_limit)

print("\nТоп-1% найбільших замовлень:")
print(
    top_orders
    .sort_values("order_value", ascending=False)
    .head(20)
)

# 8. Профіль топ-1%: категорії

top_order_ids = top_orders["order_id"]

top_categories = (
    df[df["order_id"].isin(top_order_ids)]
    .groupby("category")["item_revenue"]
    .sum()
    .reset_index()
    .sort_values("item_revenue", ascending=False)
)

print("\nКатегорії у топ-1% замовлень:")
print(top_categories)


# 9. Профіль топ-1%: країни

top_countries = (
    top_orders.groupby("ship_country")
    .agg(
        orders_count=("order_id", "count"),
        total_revenue=("order_value", "sum"),
        avg_order_value=("order_value", "mean")
    )
    .reset_index()
    .sort_values("total_revenue", ascending=False)
)

print("\nКраїни у топ-1% замовлень:")
print(top_countries)


# 10. Профіль топ-1%: сегменти

top_segments = (
    top_orders.groupby("segment")
    .agg(
        orders_count=("order_id", "count"),
        total_revenue=("order_value", "sum"),
        avg_order_value=("order_value", "mean")
    )
    .reset_index()
    .sort_values("total_revenue", ascending=False)
)

print("\nСегменти у топ-1% замовлень:")
print(top_segments)


# 11. Найменші замовлення / мікрозамовлення

small_orders_limit = orders_value["order_value"].quantile(0.01)

small_orders = orders_value[
    orders_value["order_value"] <= small_orders_limit
]

print("\nМежа найменших 1% замовлень:")
print(small_orders_limit)

print("\nНайменші замовлення:")
print(
    small_orders
    .sort_values("order_value")
    .head(20)
)

micro_orders = orders_value[
    (orders_value["order_value"] < 10) |
    (orders_value["items_count"] <= 1)
]

print("\nМікрозамовлення:")
print(
    micro_orders
    .sort_values("order_value")
    .head(20)
)

print("\nКількість мікрозамовлень:")
print(len(micro_orders))


# 12. Клієнти з одним гігантським замовленням

customer_profile = (
    orders_value.groupby("customer_id")
    .agg(
        orders_count=("order_id", "count"),
        total_spent=("order_value", "sum"),
        max_order_value=("order_value", "max"),
        avg_order_value=("order_value", "mean")
    )
    .reset_index()
)

one_big_order_customers = customer_profile[
    (customer_profile["orders_count"] == 1) &
    (customer_profile["max_order_value"] >= top_1_percent_limit)
]

print("\nКлієнти, які зробили одне гігантське замовлення і зникли:")
print(
    one_big_order_customers
    .sort_values("max_order_value", ascending=False)
)

conn.close()
