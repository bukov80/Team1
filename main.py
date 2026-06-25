import os
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations
from collections import Counter
import networkx as nx

# 1.1
# Знайти пари та трійки товарів, які найчастіше купують разом в одному замовленні.
# Розрахувати метрики support, confidence, lift. Побудувати heatmap ко-окурентності топ-20 товарів.
# Візуалізація: - Матриця зв'язків (heatmap) - Мережевий граф (network graph) топ-зв'язків
# Висновок: - Які товари варто пропонувати як cross-sell - Які товари варто розташовувати поруч на сайті

order_items = df['order_items'][['order_id', 'product_id']]
orders = order_items.groupby('order_id')['product_id'].apply(list)

# пошук пар  ==============================================================
pair_counts = Counter()
for products in orders:
    for pair in combinations(sorted(set(products)), 2):
        pair_counts[pair] += 1
# print(pair_counts)

# датафрейм
pairs_df = pd.DataFrame([
    {'product_A': a,
     'product_B': b,
     'count': c
     }
    for (a, b), c in pair_counts.items()
])

total_order_count = len(orders)

# кількість появ кожного товару)

product_counts = order_items['product_id'].value_counts()

print(product_counts)

pairs_df = pairs_df[pairs_df['count']>=8]

pairs_df['support'] = pairs_df['count']/total_order_count
pairs_df['confidence_A_B'] = (pairs_df['count']/pairs_df['product_A'].map(product_counts)).round(4)
pairs_df['confidence_B_A'] = (pairs_df['count']/pairs_df['product_B'].map(product_counts)).round(4)
pairs_df['lift'] = pairs_df['support']/(
    (pairs_df['product_A'].map(product_counts)/total_order_count) *
    (pairs_df['product_B'].map(product_counts)/total_order_count)
)
pairs_df['lift'] = pairs_df['lift'].round(4)

print(pairs_df.columns)

# print(pairs_df[['product_A', 'product_B','count', 'support', 'confidence_A-B', 'confidence_B-A','lift']].head(10))
# print('Топ - 10 пар за Lift:\n', pairs_df.sort_values(by='lift', ascending=False).head(10))
# print('Топ - 10 пар за support:\n', pairs_df.sort_values(by='support', ascending=False).head(10))

# top-20 товаров за кількістю продажів
top_20_products = product_counts.head(20).index
print('Top-20 товаров за кількістю продажів: \n', top_20_products)

# матриця зв’язків
matrix = pd.DataFrame(0, index=top_20_products, columns=top_20_products)

for (a,b), c in pair_counts.items():
    if a in top_20_products and b in top_20_products:
        matrix.loc[a,b] = c
        matrix.loc[b,a] = c

plt.figure(figsize = (10,10))
sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap Top 20 Products')
plt.show()

#  Мережевий граф (network graph) топ-зв'язків
top_pairs_30 = pairs_df.sort_values(by='lift', ascending=False).head(30)

G = nx.Graph()
for _, row in top_pairs_30.iterrows():
    G.add_edge(row['product_A'], row['product_B'], weight=row['lift'])

plt.figure(figsize = (10,10))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, edge_color='red',
        width = [d['weight'] for (_,_,d) in G.edges(data=True)])
plt.title('Top Products Association (Network Graph)')
plt.show()

# висновок по парах: =====================================================
print('Часто купують разом, товари які варто пропонувати як cross-sell (Top-10 за lift):')
print('Топ - 10 пар за Lift:\n', pairs_df.sort_values(by='lift', ascending=False)
      .head(10)[['product_A', 'product_B','count', 'lift']])

print('Товари для розташування поруч (Top-10 за support):')
print('Топ - 10 пар за support:')
print(pairs_df.sort_values(by='support', ascending=False)
      .head(10)[['product_A', 'product_B','count', 'support']])

print(pairs_df.columns.tolist())
print('Ймовірність, що з товаром купується інший товар зі списку:')
print('Топ - 10 пар за confidence:\n', pairs_df.sort_values(by='lift', ascending=False)
      .head(10)[['product_A', 'product_B','count', 'confidence_A_B', 'confidence_B_A']])

# пошук трійок: ===========================================================

pair_counts = Counter()
for products in orders:
    for pair in combinations(sorted(set(products)), 3):
        pair_counts[pair] += 1
# print(pair_counts)

# датафрейм
pairs_df = pd.DataFrame([
    {'product_A': a,
     'product_B': b,
     'product_C': c,
     'count': n
     }
    for (a, b, c), n in pair_counts.items()
])

total_order_count = len(orders)

# кількість появ кожного товару

product_counts = order_items['product_id'].value_counts()

# print('Кількість появ кожного товару: \n', product_counts)

pairs_df = pairs_df[pairs_df['count']>=2]

pairs_df['support'] = pairs_df['count']/total_order_count
pairs_df['confidence_A'] = (pairs_df['count']/pairs_df['product_A'].map(product_counts)).round(4)
pairs_df['confidence_B'] = (pairs_df['count']/pairs_df['product_B'].map(product_counts)).round(4)
pairs_df['confidence_C'] = (pairs_df['count']/pairs_df['product_C'].map(product_counts)).round(4)
pairs_df['lift'] = pairs_df['support']/(
    (pairs_df['product_A'].map(product_counts)/total_order_count) *
    (pairs_df['product_B'].map(product_counts)/total_order_count) *
    (pairs_df['product_C'].map(product_counts)/total_order_count)
)
pairs_df['lift'] = pairs_df['lift'].round(4)

# top-20 товаров за кількістю продажів
top_20_products = product_counts.head(20).index
print('Top-20 товаров за кількістю продажів: \n', top_20_products)

# матриця зв’язків
matrix = pd.DataFrame(0, index=top_20_products, columns=top_20_products)

for (a, b, c), n in pair_counts.items():
    if a in top_20_products and b in top_20_products and c in top_20_products:
        matrix.loc[a,b] += n
        matrix.loc[a,c] += n
        matrix.loc[b,a] += n
        matrix.loc[b,c] += n
        matrix.loc[c,a] += n
        matrix.loc[c,b] += n

plt.figure(figsize = (10,10))
sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap Top 20 Products')
plt.show()

#  Мережевий граф (network graph) топ-зв'язків
top_three_30 = pairs_df.sort_values(by='lift', ascending=False).head(30)

G = nx.Graph()
for _, row in top_three_30.iterrows():
    G.add_edge(row['product_A'], row['product_B'], weight=row['lift'])
    G.add_edge(row['product_A'], row['product_C'], weight=row['lift'])
    G.add_edge(row['product_B'], row['product_C'], weight=row['lift'])

plt.figure(figsize = (10,10))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, edge_color='red',
        width = [d['weight']*10 for (_,_,d) in G.edges(data=True)])
plt.title('Top Products Association (Network Graph)')
plt.show()

# висновок по трійках: ==================================================
print('Часто купують разом, товари які варто пропонувати як cross-sell (Top-10 за lift):')
print('Топ - 10 трійок за Lift:\n', pairs_df.sort_values(by='lift', ascending=False)
      .head(10)[['product_A', 'product_B','product_C','count', 'lift']])

print('Товари для розташування поруч (Top-10 за support):')
print('Топ - 10 трійок за support:')
print(pairs_df.sort_values(by='support', ascending=False)
    .head(10)[['product_A', 'product_B','product_C','count', 'support']])

# print(pairs_df.columns.tolist())
print('Ймовірність, що з товаром купується інший товар зі списку:')
print('Топ - 10 трійок за confidence:')
print(pairs_df.sort_values(by='support', ascending=False)
    .head(10)[['product_A', 'product_B','product_C','count', 'confidence_A', 'confidence_B','confidence_C']])
