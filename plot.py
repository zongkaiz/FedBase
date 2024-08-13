import matplotlib.pyplot as plt

# 数据
loss_on_g_test_data = [
    2.3028749554157257, 1.733944690823555, 0.9544915401935578, 0.6491216541472823,
    0.3915570213724859, 0.2637192090814933, 0.24961811658553779, 0.23647405985789374,
    0.2333549625494634, 0.16449341001090942, 0.19176829568989343, 0.15256663319413202,
    0.1658761263280685, 0.1680955812324537, 0.15968292063428088, 0.15388129522808594,
    0.1629080397982616, 0.1602345908742136, 0.1333404268289014, 0.16578981653606753,
    0.13684765667785542
]

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(loss_on_g_test_data, marker='o', linestyle='-', color='b', label='Loss on G Test Data')

# 设置标题和标签
plt.title('Loss on G Test Data Over Epochs')
plt.xlabel('Index')
plt.ylabel('Loss')

# 显示图例
plt.legend()


# 保存图像到文件
plt.grid(True)
plt.savefig('loss_on_g_test_data_plot.png')