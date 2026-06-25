import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect('online_store.db')

df_orders = pd.read_sql_query("SELECT * FROM orders", conn)
df_order_items = pd.read_sql_query("SELECT * FROM order_items", conn)
df_products = pd.read_sql_query("SELECT * FROM products", conn)

if "name" in df_products.columns:
    df_products = df_products.rename(columns={"name": "product_name"})
elif "product_name" not in df_products.columns:
    df_products["product_name"] = "Product " + df_products["product_id"].astype(str)

df = df_order_items.merge(df_orders, on="order_id", how="inner")
df = df.merge(
    df_products[["product_id", "product_name", "category_id"]],
    on="product_id",
    how="inner"
)

df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["year"] = df["order_date"].dt.year

df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)

df["revenue"] = df["unit_price"] * df["quantity"]

df_filtered = df[df["year"].isin([2024, 2025])].copy()

product_yearly = (
    df_filtered
    .groupby(["product_id", "product_name", "category_id", "year"])["revenue"]
    .sum()
    .unstack(fill_value=0)
)

if 2024 not in product_yearly.columns:
    product_yearly[2024] = 0.0

if 2025 not in product_yearly.columns:
    product_yearly[2025] = 0.0

product_yearly = product_yearly.rename(columns={
    2024: "rev_2024",
    2025: "rev_2025"
}).reset_index()

product_yearly["growth_rate"] = np.where(
    product_yearly["rev_2024"] > 0,
    (product_yearly["rev_2025"] - product_yearly["rev_2024"]) / product_yearly["rev_2024"],
    0
)

category_totals = (
    product_yearly
    .groupby("category_id")["rev_2025"]
    .sum()
    .reset_index(name="cat_total_rev_2025")
)

product_yearly = product_yearly.merge(category_totals, on="category_id", how="left")

product_yearly["market_share"] = (
    product_yearly["rev_2025"] / product_yearly["cat_total_rev_2025"]
).fillna(0)

growth_threshold = product_yearly["growth_rate"].median()
share_threshold = product_yearly["market_share"].median()

def classify_bcg(row):
    if row["growth_rate"] >= growth_threshold and row["market_share"] >= share_threshold:
        return "Зірки"
    elif row["growth_rate"] < growth_threshold and row["market_share"] >= share_threshold:
        return "Дійні корови"
    elif row["growth_rate"] >= growth_threshold and row["market_share"] < share_threshold:
        return "Знаки питання"
    else:
        return "Собаки"

product_yearly["quadrant"] = product_yearly.apply(classify_bcg, axis=1)

investment_needed = (
    product_yearly[product_yearly["quadrant"].isin(["Зірки", "Знаки питання"])]
    .sort_values(by="rev_2025", ascending=False)
)

print("\nТовари, що потребують інвестицій (Топ-5 за виручкою):")
print(
    investment_needed[
        ["product_name", "quadrant", "rev_2025", "growth_rate"]
    ].head(5).to_string(index=False)
)

to_remove = (
    product_yearly[product_yearly["quadrant"] == "Собаки"]
    .sort_values(by="growth_rate", ascending=True)
)

print("\nРекомендуються до виведення з асортименту (Топ-5 кандидатів):")
print(
    to_remove[
        ["product_name", "rev_2025", "growth_rate", "market_share"]
    ].head(5).to_string(index=False)
)

plt.figure(figsize=(12, 8))

colors = {
    "Зірки": "#2ca02c",
    "Дійні корови": "#1f77b4",
    "Знаки питання": "#ff7f0e",
    "Собаки": "#d62728"
}

max_rev = product_yearly["rev_2025"].max()

if max_rev > 0:
    product_yearly["bubble_size"] = product_yearly["rev_2025"] / max_rev * 1500 + 100
else:
    product_yearly["bubble_size"] = 200

sns.scatterplot(
    data=product_yearly,
    x="market_share",
    y="growth_rate",
    hue="quadrant",
    size="bubble_size",
    palette=colors,
    sizes=(100, 1600),
    alpha=0.6
)

plt.axvline(x=share_threshold, color="black", linestyle="--", alpha=0.4)
plt.axhline(y=growth_threshold, color="black", linestyle="--", alpha=0.4)

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.text(
    share_threshold + (xmax - share_threshold) * 0.3,
    growth_threshold + (ymax - growth_threshold) * 0.5,
    "ЗІРКИ",
    fontsize=16,
    fontweight="bold",
    color="green",
    alpha=0.2
)

plt.text(
    share_threshold + (xmax - share_threshold) * 0.3,
    growth_threshold - (growth_threshold - ymin) * 0.5,
    "ДІЙНІ КОРОВИ",
    fontsize=16,
    fontweight="bold",
    color="blue",
    alpha=0.2
)

plt.text(
    share_threshold - (share_threshold - xmin) * 0.7,
    growth_threshold + (ymax - growth_threshold) * 0.5,
    "ЗНАКИ ПИТАННЯ",
    fontsize=16,
    fontweight="bold",
    color="orange",
    alpha=0.2
)

plt.text(
    share_threshold - (share_threshold - xmin) * 0.7,
    growth_threshold - (growth_threshold - ymin) * 0.5,
    "СОБАКИ",
    fontsize=16,
    fontweight="bold",
    color="red",
    alpha=0.2
)

top_labelled = (
    product_yearly
    .groupby("quadrant")
    .apply(lambda x: x.nlargest(2, "rev_2025"))
    .reset_index(drop=True)
)

for _, row in top_labelled.iterrows():
    plt.text(
        row["market_share"],
        row["growth_rate"] + (ymax - ymin) * 0.02,
        row["product_name"],
        fontsize=9,
        fontweight="bold",
        ha="center",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1)
    )

plt.title("BCG матриця: класифікація товарів на квадранти (2024 → 2025)")
plt.xlabel("Частка ринку в категорії")
plt.ylabel("Темп росту виручки")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
