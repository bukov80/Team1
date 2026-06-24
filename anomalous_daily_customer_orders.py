import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

conn = sqlite3.connect("online_store.db")

query = """
SELECT
    o.order_id,
    o.customer_id,
    DATE(o.order_date) AS order_day,
    o.order_date,
    o.status,
    c.first_name,
    c.last_name,
    c.segment,
    p.name AS product_name,
    cat.name AS category,
    oi.quantity,
    oi.unit_price,
    oi.discount,
    oi.quantity * oi.unit_price * (1 - oi.discount) AS item_revenue,
    pay.method AS payment_method
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN order_items oi
    ON o.order_id = oi.order_id
JOIN products p
    ON oi.product_id = p.product_id
JOIN categories cat
    ON p.category_id = cat.category_id
LEFT JOIN payments pay
    ON o.order_id = pay.order_id
"""

df = pd.read_sql(query, conn)

df["order_date"] = pd.to_datetime(df["order_date"])
df["order_day"] = pd.to_datetime(df["order_day"])

# 1. Клієнт з найбільшою кількістю замовлень за один день

daily_orders = (
    df.groupby(
        [
            "customer_id",
            "first_name",
            "last_name",
            "segment",
            "order_day"
        ]
    )
    .agg(
        orders_count=("order_id", "nunique"),
        total_revenue=("item_revenue", "sum")
    )
    .reset_index()
)

max_orders = daily_orders["orders_count"].max()

top_clients = daily_orders[
    daily_orders["orders_count"] == max_orders
]

print("\nКлієнт(и) з найбільшою кількістю замовлень за один день:")
print(top_clients.to_string(index=False))

target_customer = top_clients.iloc[0]["customer_id"]
target_day = top_clients.iloc[0]["order_day"]

# 2. Деталі по клієнту у день аномалії

client_day_orders = df[
    (df["customer_id"] == target_customer)
    & (df["order_day"] == target_day)
]

orders_count = client_day_orders["order_id"].nunique()
total_revenue = client_day_orders["item_revenue"].sum()
total_quantity = client_day_orders["quantity"].sum()

print("\nОбраний клієнт:")
print("Customer ID:", target_customer)
print("Дата:", target_day.date())

print("\nКількість замовлень:")
print(orders_count)

print("\nЗагальна сума:")
print(round(total_revenue, 2))

print("\nЗагальна кількість товарних одиниць:")
print(total_quantity)

# 3. Усі товари клієнта у день аномалії

products = (
    client_day_orders.groupby(
        [
            "product_name",
            "category"
        ]
    )
    .agg(
        quantity=("quantity", "sum"),
        revenue=("item_revenue", "sum")
    )
    .reset_index()
    .sort_values(
        ["quantity", "revenue"],
        ascending=False
    )
)

print("\nУсі товари клієнта в день аномалії:")
print(products.to_string(index=False))

print("\nКількість різних товарів:")
print(products["product_name"].nunique())

# 4. Спосіб оплати

payments = (
    client_day_orders.groupby("payment_method")
    .agg(
        orders_count=("order_id", "nunique"),
        revenue=("item_revenue", "sum")
    )
    .reset_index()
    .sort_values("orders_count", ascending=False)
)

print("\nСпособи оплати:")
print(payments.to_string(index=False))

# 5. Повернення клієнта

returns_query = """
SELECT
    r.return_id,
    r.order_id,
    r.reason,
    r.return_date,
    r.refund_amount,
    o.customer_id
FROM returns r
JOIN orders o
    ON r.order_id = o.order_id
WHERE o.customer_id = ?
"""

returns_df = pd.read_sql(
    returns_query,
    conn,
    params=(int(target_customer),)
)

print("\nПовернення клієнта:")
if len(returns_df) > 0:
    print(returns_df.to_string(index=False))

    reasons = (
        returns_df.groupby("reason")
        .agg(
            returns_count=("return_id", "count"),
            refund_amount=("refund_amount", "sum")
        )
        .reset_index()
        .sort_values("returns_count", ascending=False)
    )

    print("\nПричини повернень:")
    print(reasons.to_string(index=False))
else:
    print("Повернень не знайдено")

# 6. Історія замовлень клієнта по днях

history = (
    df[df["customer_id"] == target_customer]
    .groupby("order_day")
    .agg(
        orders_count=("order_id", "nunique"),
        revenue=("item_revenue", "sum")
    )
    .reset_index()
    .sort_values("order_day")
)

print("\nІсторія замовлень клієнта:")
print(history.to_string(index=False))

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=history,
    x="order_day",
    y="orders_count",
    marker="o"
)

plt.axvline(
    target_day,
    linestyle="--",
    label="Аномальний день"
)

plt.title("Історія замовлень клієнта по днях")
plt.xlabel("Дата")
plt.ylabel("Кількість замовлень")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 7. Висновок: оптовий клієнт чи фрод

total_orders = df[
    df["customer_id"] == target_customer
]["order_id"].nunique()

active_days = df[
    df["customer_id"] == target_customer
]["order_day"].nunique()

returns_count = len(returns_df)

print("\nПоказники для висновку:")
print("Усього замовлень:", total_orders)
print("Активних днів:", active_days)
print("Повернень:", returns_count)
print("Revenue у день аномалії:", round(total_revenue, 2))

print("\nВисновок:")

if active_days > 5 and returns_count == 0:
    print("Ймовірно справжній оптовий клієнт: є активність у різні дні та немає повернень.")
elif active_days <= 2 and returns_count > 0:
    print("Є ризик фроду: багато замовлень сконцентровані в короткий період і є повернення.")
elif orders_count >= 5 and active_days <= 2:
    print("Підозріла активність: багато замовлень за один день, але мало активних днів.")
else:
    print("Ознак явного фроду недостатньо, але клієнта варто перевірити вручну.")

conn.close()