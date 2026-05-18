import matplotlib.pyplot as plt

# 数据
clusters = [64, 128, 192, 256, 320, 384, 448, 512]
mAP = [91.32, 92.87, 93.22, 93.84, 93.94, 94.32, 94.01, 94.33]
recall = [94.60, 96.00, 96.54, 97.14, 97.32, 97.80, 97.55, 97.71]
speedup = [20.7, 21.3, 21.8, 19.7, 17.9, 16.6, 15.0, 13.9]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14  # 设置为四号字体 (14pt)

# 图1
plt.figure(figsize=(7, 5))
plt.plot(clusters, mAP, marker='o', linewidth=2, label='mAP')
plt.plot(clusters, recall, marker='s', linewidth=2, label='Recall')
plt.xlabel('聚类中心数')
plt.ylabel('指标值/%')
plt.title('聚类中心数对匹配精度的影响')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(r'results\accuracy_curve.png', dpi=300)
plt.close()

# 图2
plt.figure(figsize=(7, 5))
plt.plot(clusters, speedup, marker='^', linewidth=2, color='tab:green')
plt.xlabel('聚类中心数')
plt.ylabel('加速比/倍')
plt.title('聚类中心数对检索加速比的影响')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(r'results\speedup_curve.png', dpi=300)
plt.close()

print("图片已保存到")
