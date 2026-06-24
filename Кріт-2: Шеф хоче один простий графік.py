# Шеф хоче **один простий графік** для презентації на раді директорів — «які промоакції дають найбільший обсяг продажів?».  
# 1. Згрупувати order_items по `promotion_id`, порахувати сумарний revenue та сумарний discount 
# 2. Об'єднати з таблицею `promotions` (назва акції, discount_pct) 
# 3. Побудувати **horizontal bar chart** — revenue по промоакціях (тільки ТОП-5) 
# 4. На цьому ж графіку показати **discount_pct** кольором 
# 5. Вивести таблицю з назвою акції, revenue, знижкою

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "online_store.db"

conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    p.name AS promotion_name,
    p.discount_pct,

    SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue,

    SUM(oi.quantity * oi.unit_price * oi.discount) AS total_discount

FROM order_items oi
JOIN promotions p
    ON oi.promotion_id = p.promotion_id

GROUP BY
    p.promotion_id,
    p.name,
    p.discount_pct
"""

df = pd.read_sql(query, conn)
conn.close()

# ТОП-5 промоакцій за revenue
top5 = (
    df.sort_values("revenue", ascending=False)
      .head(5)
)

print("\nTOP-5 promotions by revenue:")
print(
    top5[["promotion_name", "revenue", "discount_pct"]]
)

# Графік
plt.figure(figsize=(10, 6))

sns.barplot(
    data=top5,
    x="revenue",
    y="promotion_name",
    hue="discount_pct",
    dodge=False
)

plt.title("Top 5 Promotions by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Promotion")
plt.legend(title="Discount %")

plt.tight_layout()
plt.show()

