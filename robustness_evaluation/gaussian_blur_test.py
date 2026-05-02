import os
from evaluate_robustness_fundefine import evaluate_robustness

if os.path.exists('./classes/classes.txt'):
    with open('classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"成功加载类别: {len(class_names)} 个")
else:
    # 容错：如果文件不存在则根据训练目录推断
    class_names = sorted(os.listdir('./data/train'))
    print("classes.txt 不存在，从数据目录推断类别名称。")

best_model_path = './pth/ablation_experiment/plant_disease_mobilenet_class_weights.pth'
test_dir = './data/test' 
# 1. 定义模糊强度 (必须是奇数)
blur_intensities = [3, 5, 7, 9, 11]

# 2. 执行高斯模糊鲁棒性评估
blur_results = evaluate_robustness(
    model_path=best_model_path,
    test_dir=test_dir,
    class_names=class_names,
    intensities=blur_intensities,
    mode='blur'
)

print("\n高斯模糊测试完成。")
print("准确率分布 (核大小: 准确率):", blur_results)