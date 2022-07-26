import numpy as np
import matplotlib.pyplot as plt

shops = ["A", "B", "C", "D", "E", "F"]
sales_product_1 = [100, 85, 56, 42, 72, 15]
sales_product_2 = [50, 120, 65, 85, 25, 55]
sales_product_3 = [20, 35, 45, 27, 55, 65]

# 创建分组柱状图，需要自己控制x轴坐标
xticks = np.arange(len(shops))

fig, ax = plt.subplots(figsize=(10, 7))
# 所有门店第一种产品的销量，注意控制柱子的宽度，这里选择0.25
ax.bar(xticks, sales_product_1, width=0.25, label="Product_1", color="red")
# 所有门店第二种产品的销量，通过微调x轴坐标来调整新增柱子的位置
ax.bar(xticks + 0.25, sales_product_2, width=0.25, label="Product_2", color="blue")
# 所有门店第三种产品的销量，继续微调x轴坐标调整新增柱子的位置
ax.bar(xticks + 0.5, sales_product_3, width=0.25, label="Product_3", color="green")

ax.set_title("Grouped Bar plot", fontsize=15)
ax.set_xlabel("Shops")
ax.set_ylabel("Product Sales")
ax.legend()

# 最后调整x轴标签的位置
ax.set_xticks(xticks + 0.25)
ax.set_xticklabels(shops)