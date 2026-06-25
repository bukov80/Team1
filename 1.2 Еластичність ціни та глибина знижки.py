#1.2 Еластичність ціни та глибина знижки:
#Для кожної категорії проаналізувати, як глибина знижки (discount %) впливає на кількість проданих одиниць та маржу.
#Завдання:
#- Розрахувати коефіцієнт кореляції між discount % та quantity по категоріях
#- Визначити оптимальний діапазон знижки, що максимізує revenue без втрати margin
#Візуалізація: - Scatter plot (discount % vs quantity) з лінією регресії - Facet по категоріях


import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DB_PATH = "online_store.db"

conn = sqlite3.connect(DB_PATH)

query = """
SELECT 
    c.name AS category,
    oi.quantity,
    oi.unit_price,
    oi.discount AS discount,
    p.cost,
    o.status
FROM order_items oi
JOIN orders o 
    ON oi.order_id = o.order_id
JOIN products p 
    ON oi.product_id = p.product_id
JOIN categories c 
    ON p.category_id = c.category_id
WHERE o.status = 'completed'
"""

df = pd.read_sql(query, conn)
conn.close()

# Підготовка даних

df["discount_percent"] = df["discount"] * 100

df["revenue"] = (
    df["quantity"] *
    df["unit_price"] *
    (1 - df["discount"])
)

df["margin"] = (
    df["unit_price"] *
    (1 - df["discount"]) -
    df["cost"]
) * df["quantity"]

df["margin_rate"] = df["margin"] / df["revenue"]

print("\nПерші рядки:")
print(df.head())

print("\nОпис знижок:")
print(df["discount_percent"].describe())

# 1. Кореляція discount % та quantity по категоріях

correlation_by_category = (
    df.groupby("category")
      .apply(
          lambda x: x["discount_percent"].corr(x["quantity"]),
          include_groups=False
      )
      .reset_index(name="discount_quantity_corr")
      .sort_values("discount_quantity_corr", ascending=False)
)

print("\nКореляція між discount % та quantity по категоріях:")
print(correlation_by_category)

# 2. Діапазони знижок

bins = [0, 5, 10, 15, 20, 25, 50, 100]
labels = [
    "0-5%",
    "5-10%",
    "10-15%",
    "15-20%",
    "20-25%",
    "25-50%",
    "50-100%"
]

df["discount_range"] = pd.cut(
    df["discount_percent"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)

# 3. Аналіз revenue та margin по категоріях і знижках

discount_analysis = (
    df.groupby(["category", "discount_range"], observed=True)
      .agg(
          revenue=("revenue", "sum"),
          margin=("margin", "sum"),
          quantity=("quantity", "sum"),
          avg_margin=("margin", "mean"),
          avg_margin_rate=("margin_rate", "mean"),
          orders_count=("quantity", "count")
      )
      .reset_index()
)

discount_analysis["margin_rate_total"] = (
    discount_analysis["margin"] / discount_analysis["revenue"]
)

print("\nАналіз revenue та margin по діапазонах знижок:")
print(discount_analysis)

# 4. Перевірка збиткових діапазонів

negative_margin = discount_analysis[
    discount_analysis["margin"] < 0
]

print("\nДіапазони, де margin від'ємна:")
print(negative_margin)

# 5. Оптимальний діапазон знижки

# Логіка:
# 1. revenue має бути високим
# 2. margin має бути позитивною
# 3. margin_rate_total має бути не нижче 20%

safe_discount_analysis = discount_analysis[
    (discount_analysis["margin"] > 0) &
    (discount_analysis["margin_rate_total"] >= 0.20)
]

optimal_discount = (
    safe_discount_analysis
    .sort_values(["category", "revenue"], ascending=[True, False])
    .groupby("category")
    .head(1)
    .reset_index(drop=True)
)

print("\nОптимальний діапазон знижки по категоріях:")
print(
    optimal_discount[
        [
            "category",
            "discount_range",
            "revenue",
            "margin",
            "margin_rate_total",
            "quantity",
            "orders_count"
        ]
    ]
)


# 6. Scatter plot: discount % vs quantity з регресією

sns.set_theme(style="whitegrid")

g = sns.lmplot(
    data=df,
    x="discount_percent",
    y="quantity",
    col="category",
    col_wrap=4,
    height=3,
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"}
)

g.set_axis_labels("Discount, %", "Quantity")
g.set_titles("{col_name}")

plt.tight_layout()
plt.show()

# 7. Аналіз margin по діапазонах знижок


plt.figure(figsize=(12, 6))

sns.barplot(
    data=discount_analysis,
    x="discount_range",
    y="margin_rate_total",
    hue="category"
)

plt.axhline(
    0.20,
    linestyle="--",
    color="red",
    label="Minimum safe margin rate 20%"
)

plt.title("Margin rate по діапазонах знижок")
plt.xlabel("Discount range")
plt.ylabel("Margin rate")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 8. Revenue по діапазонах знижок

plt.figure(figsize=(12, 6))

sns.barplot(
    data=discount_analysis,
    x="discount_range",
    y="revenue",
    hue="category"
)

plt.title("Revenue по діапазонах знижок")
plt.xlabel("Discount range")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 9. Текстові висновки

print("\nВисновки по оптимальних знижках:")

for _, row in optimal_discount.iterrows():
    print(
        f"Категорія: {row['category']} | "
        f"Оптимальний діапазон: {row['discount_range']} | "
        f"Revenue: {row['revenue']:.2f} | "
        f"Margin: {row['margin']:.2f} | "
        f"Margin rate: {row['margin_rate_total']:.2%}"
    )


plt.tight_layout()
plt.show()
