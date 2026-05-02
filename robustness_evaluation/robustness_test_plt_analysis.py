import matplotlib.pyplot as plt
from gaussian_blur_test import blur_results
from gaussian_noise_test import noise_results

# 提取结果数据
noise_x = list(noise_results.keys())
noise_y = list(noise_results.values())

blur_x = list(blur_results.keys())
blur_y = list(blur_results.values())

plt.figure(figsize=(14, 6))

# 绘制高斯噪声图
plt.subplot(1, 2, 1)
plt.plot(noise_x, noise_y, marker='o', color='orange', linewidth=2)
plt.title('Accuracy vs Gaussian Noise Intensity')
plt.xlabel('Noise Std Dev')
plt.ylabel('Test Accuracy')
plt.grid(True, linestyle='--')
plt.ylim(0.9, 1.01)

# 绘制高斯模糊图
plt.subplot(1, 2, 2)
plt.plot(blur_x, blur_y, marker='s', color='blue', linewidth=2)
plt.title('Accuracy vs Gaussian Blur Kernel Size')
plt.xlabel('Kernel Size (px)')
plt.ylabel('Test Accuracy')
plt.grid(True, linestyle='--')
plt.ylim(0.9, 1.01)

plt.tight_layout()
plt.show()