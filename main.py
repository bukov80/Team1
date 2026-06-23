import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect('online_store.db')

df_orders = pd.read_sql_query("SELECT * FROM orders", conn)
df_order_items = pd.read_sql_query("SELECT * FROM order_items", conn)
df_products = pd.read_sql_query("SELECT * FROM products", conn)

if 'name' in df_products.columns:
    df_products = df_products.rename(columns={'name': 'product_name'})
elif 'product_name' not in df_products.columns:
    df_products['product_name'] = 'Product ' + df_products['product_id'].astype(str)

df = df_order_items.merge(df_orders, on='order_id', how='inner')
df = df.merge(df_products[['product_id', 'product_name', 'category_id']], on='product_id', how='inner')

df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df = df.dropna(subset=['order_date'])
df['revenue'] = pd.to_numeric(df['unit_price']) * pd.to_numeric(df['quantity'])

df_daily = df.groupby('order_date')['revenue'].sum().reset_index()
df_daily = df_daily.set_index('order_date').asfreq('D', fill_value=0)

df_daily['trend'] = df_daily['revenue'].rolling(window=7, center=True, min_periods=1).mean()

df_daily['detrended'] = df_daily['revenue'] - df_daily['trend']

df_daily['weekday'] = df_daily.index.day_name()

weekday_seasonality = df_daily.groupby('weekday')['detrended'].mean()
df_daily = df_daily.merge(weekday_seasonality.rename('seasonal'), left_on='weekday', right_index=True, how='left')

df_daily['residual'] = df_daily['revenue'] - df_daily['trend'] - df_daily['seasonal']

std_resid = df_daily['residual'].std()
mean_resid = df_daily['residual'].mean()
df_daily['is_anomaly'] = (df_daily['residual'] > mean_resid + 3 * std_resid) | (df_daily['residual'] < mean_resid - 3 * std_resid)
anomalies = df_daily[df_daily['is_anomaly']]

df_daily['year'] = df_daily.index.year
df_daily['month'] = df_daily.index.month_name()

months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

df_sports = df[df['category_id'].astype(str).str.contains('sports|10|4', case=False, na=False)]
df_electronics = df[df['category_id'].astype(str).str.contains('electronics|electron|1|5', case=False, na=False)]

sports_monthly = df_sports.groupby(df_sports['order_date'].dt.month_name())['revenue'].sum().reindex(months_order).fillna(0)
electronics_monthly = df_electronics.groupby(df_electronics['order_date'].dt.month_name())['revenue'].sum().reindex(months_order).fillna(0)

weekday_avg = df_daily.groupby('weekday')['revenue'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
print("\n Средняя выручка по дням недели:")
print(weekday_avg.to_string())

print("\n Выявленные аномальные дни (Топ-5 отклонений):")
if not anomalies.empty:
    print(anomalies[['revenue', 'residual']].sort_values(by='residual', ascending=False).head(5).to_string())
else:
    print("Явных изолированных аномалий не обнаружено.")

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(df_daily.index, df_daily['revenue'], color='#1f77b4', alpha=0.6, label='Фактическая выручка ($)')
axes[0].set_title('Разложение временного ряда выручки (Time Series Decomposition)', fontsize=14, pad=10)
axes[0].legend(loc='upper left')
axes[0].grid(True, linestyle=':', alpha=0.5)

axes[1].plot(df_daily.index, df_daily['trend'], color='#d62728', linewidth=2, label='Годовой тренд (Trend)')
axes[1].legend(loc='upper left')
axes[1].grid(True, linestyle=':', alpha=0.5)

axes[2].plot(df_daily.index, df_daily['seasonal'], color='#2ca02c', label='Недельная сезонность (Seasonal)')
axes[2].legend(loc='upper left')
axes[2].grid(True, linestyle=':', alpha=0.5)

axes[3].scatter(df_daily.index, df_daily['residual'], color='#9467bd', alpha=0.5, s=12, label='Остатки (Residual / Шум)')
if not anomalies.empty:
    axes[3].scatter(anomalies.index, anomalies['residual'], color='red', s=35, edgecolor='black', label='Выявленные аномалии (Пики)')
axes[3].legend(loc='upper left')
axes[3].grid(True, linestyle=':', alpha=0.5)
axes[3].set_xlabel('Дата заказа')

plt.tight_layout()
plt.savefig('revenue_decomposition_linear.png', dpi=300)
plt.close()

latest_year = int(df_daily['year'].max())
df_year = df_daily[df_daily['year'] == latest_year].copy()

df_year['week_of_year'] = df_year.index.isocalendar().week
df_year['day_of_week_num'] = df_year.index.dayofweek

cal_pivot = df_year.pivot_table(index='day_of_week_num', columns='week_of_year', values='revenue', aggfunc='sum').fillna(0)

plt.figure(figsize=(15, 4))
days_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

sns.heatmap(
    cal_pivot,
    cmap='YlOrRd',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Дневная выручка ($)', 'orientation': 'horizontal', 'pad': 0.2}
)

plt.title(f'Календарный Heatmap выручки за {latest_year} год (Дни недели × Недели года)', fontsize=14, pad=15)
plt.ylabel('День недели')
plt.xlabel('Порядковый номер недели в году')
plt.yticks(np.arange(7) + 0.5, days_labels, rotation=0)
plt.tight_layout()

plt.savefig('revenue_calendar_heatmap.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
x_indexes = np.arange(len(months_order))
width = 0.35

plt.bar(x_indexes - width/2, sports_monthly, width, label='Sports', color='orange', alpha=0.8)
plt.bar(x_indexes + width/2, electronics_monthly, width, label='Electronics', color='teal', alpha=0.8)

plt.title('Проверка сезонности категорий товаров (Продажи по месяцам)')
plt.xticks(x_indexes, [m[:3] for m in months_order])
plt.xlabel('Месяц')
plt.ylabel('Выручка ($)')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('category_seasonality_comparison.png', dpi=300)
plt.close()

conn.close()
