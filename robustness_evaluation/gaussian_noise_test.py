import os
from evaluate_robustness_fundefine import evaluate_robustness

noise_intensities = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]

# 3. 指定消融实验中表现最好的模型
if os.path.exists('./classes/classes.txt'):
    with open('./classes/classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"成功加载类别: {len(class_names)} 个")
else:
    # 容错：如果文件不存在则根据训练目录推断
    class_names = sorted(os.listdir('./data/train'))
    print("classes.txt 不存在，从数据目录推断类别名称。")

best_model_path = './pth/ablation_experiment/plant_disease_mobilenet_class_weights.pth'
test_dir = './data/test'

# 4. 执行鲁棒性评估
noise_results = evaluate_robustness(
    model_path=best_model_path,
    test_dir=test_dir,
    class_names=class_names,
    intensities=noise_intensities,
    mode='noise'
)

print("\n高斯噪声测试完成。")
print("准确率分布:", noise_results)