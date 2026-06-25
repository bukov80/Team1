import os
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import combinations
from collections import Counter
import networkx as nx

DB_PATH = "online_store.db"

conn = sqlite3.connect(DB_PATH)

order_items = pd.read_sql("""
    SELECT order_id, product_id
    FROM order_items
""", conn)

conn.close()

orders = order_items.groupby("order_id")["product_id"].apply(list)

# =========================
# Пошук пар товарів
# =========================

pair_counts = Counter()

for products in orders:
    unique_products = sorted(set(products))
    for pair in combinations(unique_products, 2):
        pair_counts[pair] += 1

pairs_df = pd.DataFrame([
    {
        "product_A": a,
        "product_B": b,
        "count": count
    }
    for (a, b), count in pair_counts.items()
])

total_order_count = len(orders)
product_counts = order_items["product_id"].value_counts()

pairs_df = pairs_df[pairs_df["count"] >= 8].copy()

pairs_df["support"] = pairs_df["count"] / total_order_count

pairs_df["confidence_A_B"] = (
    pairs_df["count"] / pairs_df["product_A"].map(product_counts)
).round(4)

pairs_df["confidence_B_A"] = (
    pairs_df["count"] / pairs_df["product_B"].map(product_counts)
).round(4)

pairs_df["lift"] = pairs_df["support"] / (
    (pairs_df["product_A"].map(product_counts) / total_order_count) *
    (pairs_df["product_B"].map(product_counts) / total_order_count)
)

pairs_df["lift"] = pairs_df["lift"].round(4)

print("Топ-10 пар за Lift:")
print(
    pairs_df.sort_values("lift", ascending=False)
    .head(10)[["product_A", "product_B", "count", "support", "confidence_A_B", "confidence_B_A", "lift"]]
)

print("\nТоп-10 пар за Support:")
print(
    pairs_df.sort_values("support", ascending=False)
    .head(10)[["product_A", "product_B", "count", "support"]]
)

# Heatmap top-20 товарів

top_20_products = product_counts.head(20).index

matrix = pd.DataFrame(0, index=top_20_products, columns=top_20_products)

for (a, b), count in pair_counts.items():
    if a in top_20_products and b in top_20_products:
        matrix.loc[a, b] = count
        matrix.loc[b, a] = count

plt.figure(figsize=(12, 10))
sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Heatmap ко-окупності Top-20 товарів")
plt.xlabel("Product ID")
plt.ylabel("Product ID")
plt.tight_layout()
plt.show()

# Network graph top-30 пар

top_pairs_30 = pairs_df.sort_values("lift", ascending=False).head(30)

G = nx.Graph()

for _, row in top_pairs_30.iterrows():
    G.add_edge(
        row["product_A"],
        row["product_B"],
        weight=row["lift"]
    )

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.5, seed=42)

edge_widths = [max(1, d["weight"] / 2) for _, _, d in G.edges(data=True)]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=700,
    font_size=8,
    edge_color="red",
    width=edge_widths
)

plt.title("Network Graph топ-зв'язків між товарами")
plt.show()

print("\nВисновок по парах:")
print("Товари з найбільшим lift варто пропонувати як cross-sell.")
print("Товари з найбільшим support варто розміщувати поруч на сайті.")


# =========================
# Пошук трійок товарів
# =========================

triple_counts = Counter()

for products in orders:
    unique_products = sorted(set(products))
    for triple in combinations(unique_products, 3):
        triple_counts[triple] += 1

triples_df = pd.DataFrame([
    {
        "product_A": a,
        "product_B": b,
        "product_C": c,
        "count": count
    }
    for (a, b, c), count in triple_counts.items()
])

triples_df = triples_df[triples_df["count"] >= 2].copy()

triples_df["support"] = triples_df["count"] / total_order_count

triples_df["confidence_A"] = (
    triples_df["count"] / triples_df["product_A"].map(product_counts)
).round(4)

triples_df["confidence_B"] = (
    triples_df["count"] / triples_df["product_B"].map(product_counts)
).round(4)

triples_df["confidence_C"] = (
    triples_df["count"] / triples_df["product_C"].map(product_counts)
).round(4)

triples_df["lift"] = triples_df["support"] / (
    (triples_df["product_A"].map(product_counts) / total_order_count) *
    (triples_df["product_B"].map(product_counts) / total_order_count) *
    (triples_df["product_C"].map(product_counts) / total_order_count)
)

triples_df["lift"] = triples_df["lift"].round(4)

print("\nТоп-10 трійок за Lift:")
print(
    triples_df.sort_values("lift", ascending=False)
    .head(10)[["product_A", "product_B", "product_C", "count", "support", "lift"]]
)

print("\nТоп-10 трійок за Support:")
print(
    triples_df.sort_values("support", ascending=False)
    .head(10)[["product_A", "product_B", "product_C", "count", "support"]]
)

print("\nТоп-10 трійок за Confidence:")
print(
    triples_df.sort_values("support", ascending=False)
    .head(10)[["product_A", "product_B", "product_C", "count", "confidence_A", "confidence_B", "confidence_C"]]
)

# Network graph top-30 трійок

top_triples_30 = triples_df.sort_values("lift", ascending=False).head(30)

G = nx.Graph()

for _, row in top_triples_30.iterrows():
    G.add_edge(row["product_A"], row["product_B"], weight=row["lift"])
    G.add_edge(row["product_A"], row["product_C"], weight=row["lift"])
    G.add_edge(row["product_B"], row["product_C"], weight=row["lift"])

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.5, seed=42)

edge_widths = [max(1, d["weight"] / 2) for _, _, d in G.edges(data=True)]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=700,
    font_size=8,
    edge_color="red",
    width=edge_widths
)

plt.title("Network Graph топ-зв'язків між трійками товарів")
plt.show()

print("\nВисновок по трійках:")
print("Трійки з найбільшим lift можна використовувати для пакетних пропозицій.")
print("Трійки з найбільшим support показують найчастіші комбінації товарів.")
