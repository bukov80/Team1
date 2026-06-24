import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

conn = sqlite3.connect('online_store.db')

df_orders = pd.read_sql_query("SELECT * FROM orders", conn)
df_order_items = pd.read_sql_query("SELECT * FROM order_items", conn)
df_customers = pd.read_sql_query("SELECT * FROM customers", conn)
df_products = pd.read_sql_query("SELECT * FROM products", conn)

if 'name' in df_products.columns:
    df_products = df_products.rename(columns={'name': 'product_name'})

df_order_items['unit_price'] = pd.to_numeric(df_order_items['unit_price']).fillna(0)
df_order_items['quantity'] = pd.to_numeric(df_order_items['quantity']).fillna(0)
df_order_items['discount'] = pd.to_numeric(df_order_items['discount']).fillna(0)

df_order_items['item_total'] = df_order_items['unit_price'] * df_order_items['quantity'] * (1 - df_order_items['discount'])

df_order_values = df_order_items.groupby('order_id')['item_total'].sum().reset_index(name='order_value')

df_orders_enriched = df_orders.merge(df_order_values, on='order_id', how='inner')
df_orders_enriched = df_orders_enriched.merge(df_customers, on='customer_id', how='inner')

df_orders_enriched['full_name'] = df_orders_enriched['first_name'] + ' ' + df_orders_enriched['last_name']

values = df_orders_enriched['order_value']

q1 = values.quantile(0.25)
q3 = values.quantile(0.75)
iqr = q3 - q1
upper_bound_iqr = q3 + 1.5 * iqr
outliers_iqr = df_orders_enriched[values > upper_bound_iqr]

mean_val = values.mean()
std_val = values.std()
df_orders_enriched['z_score'] = np.abs((values - mean_val) / std_val)
outliers_z = df_orders_enriched[df_orders_enriched['z_score'] > 3]

print("=== СТАТИСТИЧЕСКИЙ АНАЛИЗ И ВЫЯВЛЕНИЕ ВЫБРОСОВ ===")
print(f"Всего заказов в базе: {len(df_orders_enriched)}")
print(f"Выбросов по методу IQR (чеки > {upper_bound_iqr:.2f}$): {len(outliers_iqr)}")
print(f"Выбросов по методу Z-score (сильные аномалии): {len(outliers_z)}")

top_1_percent_threshold = values.quantile(0.99)
df_top_1 = df_orders_enriched[values >= top_1_percent_threshold]

df_top_1_items = df_order_items[df_order_items['order_id'].isin(df_top_1['order_id'])].merge(df_products, on='product_id', how='inner')
top_categories = df_top_1_items.groupby('category_id')['item_total'].sum().sort_values(ascending=False).reset_index()

print(f"\n ПРОФАЙЛ ТОП-1% НАИБОЛЬШИХ ЗАКАЗОВ (Чеки >= {top_1_percent_threshold:.2f}$):")
print(f"• Top стран доставки:\n{df_top_1['ship_country'].value_counts().head(3).to_string()}")
print(f"• Топ сегментов клиентов:\n{df_top_1['segment'].value_counts().head(3).to_string()}")
print(f"• Топ категорий товара по выручке в крупных чеках:\n{top_categories.head(3).to_string(index=False, float_format='%.2f')}")

df_micro = df_orders_enriched[values < values.quantile(0.01)]
print("\n АНАЛИЗ МИКРОЗАКАЗОВ (Топ-1% снизу):")
print(f"• Минимальный чек в магазине: {values.min():.2f}$")
print(f"• Средний чек micro-заказов: {df_micro['order_value'].mean():.2f}$")

customer_order_counts = df_orders_enriched.groupby('customer_id').size().reset_index(name='total_customer_orders')
df_top_1_with_counts = df_top_1.merge(customer_order_counts, on='customer_id', how='inner')
one_time_giant_buyers = df_top_1_with_counts[df_top_1_with_counts['total_customer_orders'] == 1]

print("\n КЛИЕНТЫ С ОДНИМ ГИГАНТСКИМ ЗАКАЗОМ (И исчезли):")
print(f"• Количество таких клиентов: {len(one_time_giant_buyers)}")
if not one_time_giant_buyers.empty:
    print(one_time_giant_buyers[['customer_id', 'full_name', 'order_value', 'ship_country']].head(5).to_string(index=False, float_format='%.2f'))

plt.figure(figsize=(10, 6))
sns.histplot(values[values <= top_1_percent_threshold], bins=40, kde=True, color='#1f77b4', alpha=0.7)

plt.axvline(q1, color='orange', linestyle='--', linewidth=2, label=f'Q1 (25%): {q1:.1f}$')
plt.axvline(values.median(), color='red', linestyle='-', linewidth=2, label=f'Median (50%): {values.median():.1f}$')
plt.axvline(q3, color='green', linestyle='--', linewidth=2, label=f'Q3 (75%): {q3:.1f}$')
plt.axvline(upper_bound_iqr, color='purple', linestyle=':', linewidth=2, label=f'Граница IQR: {upper_bound_iqr:.1f}$')

plt.title('Распределение стоимости заказов (Гистограмма с квартилями)', fontsize=14, pad=15)
plt.xlabel('Стоимость заказа ($)', fontsize=12)
plt.ylabel('Количество заказов', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('orders_distribution_histogram.png', dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_orders_enriched, x='segment', y='order_value', palette='Set2', hue='segment', legend=False)

plt.title('Диаграмма "Ящик с усами" (Box Plot) по сегментам покупателей', fontsize=14, pad=15)
plt.xlabel('Сегмент клиента', fontsize=12)
plt.ylabel('Стоимость заказа ($)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)

plt.ylim(0, upper_bound_iqr * 2.5)
plt.tight_layout()
plt.savefig('orders_value_boxplot.png', dpi=300)
plt.close()

conn.close()
