# PlantDisease：基于 MobileNetV3 的植物叶片病害分类

> 基于 **PyTorch + MobileNetV3-Small** 的 15 类植物叶片病害识别项目，目标是在 Android 端高效部署。
> 项目按"分步骤手动运行"组织，每个脚本独立可控，便于消融对比与论文/报告分析，**不是一键流水线**。

---

## 目录

1. [项目概览](#一项目概览)
2. [当前进度](#二当前进度)
3. [目录结构](#三目录结构)
4. [数据集](#四数据集)
5. [环境依赖](#五环境依赖)
6. [模型与训练配置](#六模型与训练配置)
7. [运行流程（分步骤）](#七运行流程分步骤)
8. [实验方案](#八实验方案)
9. [部署与 Android App](#九部署与-android-app)
10. [性能评估指标](#十性能评估指标)
11. [关键脚本对照表](#十一关键脚本对照表)
12. [运行注意事项](#十二运行注意事项)
13. [许可与致谢](#十三许可与致谢)

---

## 一、项目概览

| 项 | 内容 |
|---|---|
| **目标** | 基于 MobileNetV3-Small 的植物病害识别系统，落地 Android 端 |
| **训练框架** | PyTorch |
| **部署框架** | TensorFlow Lite（`.pth → .onnx → .tflite`） |
| **任务类型** | 图像多分类（15 类） |
| **目标作物** | 辣椒 (Pepper)、马铃薯 (Potato)、番茄 (Tomato) |
| **当前阶段** | 模型训练已完成，进入消融分析、鲁棒性评估与 App 开发阶段 |

---

## 二、当前进度

* **网络架构**：MobileNetV3-small（迁移学习）
* **训练 Epochs**：15
* **收敛性**：首轮验证集准确率即达 **92.42%**，第 13 轮达到峰值
* **最佳验证准确率**：**99.35%**
* **最优权重**：已保存（待转换为 TFLite）

---

## 三、目录结构

```
PlantDisease/
├── README.md
├── CLAUDE.md                         # 协作行为约束（与代码无关）
├── requirements.txt                  # PyTorch 2.10 + CUDA 12.8 等依赖
│
├── classes/
│   └── classes.txt                   # 训练后写入的 15 个类别名（顺序敏感）
│
├── cuda_acailable.py                 # 检测当前 CUDA 设备
├── download_data.py                  # 从 Kaggle 下载并按 7:1.5:1.5 划分数据
├── train_model.py                    # 主训练脚本（完整版：增强+类别权重+预训练）
├── train_model_plt_analysis.py       # 主训练曲线绘制
│
├── data/                             # 划分后的数据（脚本生成）
│   ├── train/<class>/...
│   ├── val/<class>/...
│   └── test/<class>/...
│
├── plantdisease/PlantVillage/        # Kaggle 原始数据（脚本下载）
│
├── pth/
│   └── ablation_experiment/          # 消融实验权重存档
│       ├── plant_disease_mobilenet_baseline.pth
│       ├── plant_disease_mobilenet_augmentation.pth
│       ├── plant_disease_mobilenet_class_weights.pth
│       └── plant_disease_mobilenet_no_pretrained.pth
│
├── ablation_experiment/              # 消融实验
│   ├── ablation_train_model.py             # 通用训练函数（3 个开关）
│   ├── baseline_ablation_train_model.py    # 变体 1：纯预训练
│   ├── augmentation_ablation_train_model.py# 变体 2：+ 数据增强
│   ├── class_weights_ablation_train_model.py# 变体 3：+ 类别权重
│   ├── no_pretrained_ablation_train_model.py# 变体 4：去掉预训练
│   ├── abliation_model_evaluation.py       # 4 个模型的测试集评估
│   └── ablation_plt_analysis.py            # 4 个模型的训练曲线对比图
│
├── robustness_evaluation/            # 鲁棒性评估
│   ├── evaluate_robustness_fundefine.py    # 评估函数定义（noise / blur）
│   ├── gaussian_noise_test.py              # 高斯噪声测试
│   ├── gaussian_blur_test.py               # 高斯模糊测试
│   └── robustness_test_plt_analysis.py     # 鲁棒性曲线绘图
│
└── png/                              # 输出的可视化图片
    ├── train_model_plt_analysis.png
    ├── ablation_history_comparison.png
    └── robustness_test_plt_analysis.png
```

---

## 四、数据集

* **来源**：[Kaggle - PlantVillage by emmarex](https://www.kaggle.com/datasets/emmarex/plantdisease)
* **样本规模**：共 **20,638 张**图像
* **类别总数**：**15 类**（健康 + 多种常见病害）
* **划分比例**：训练 70% / 验证 15% / 测试 15%（每个类别内部独立 shuffle 后切分）

### 类别清单（顺序与 `classes/classes.txt` 一致）

| # | 类别 |
|---|------|
| 1 | Pepper__bell___Bacterial_spot |
| 2 | Pepper__bell___healthy |
| 3 | Potato___Early_blight |
| 4 | Potato___Late_blight |
| 5 | Potato___healthy |
| 6 | Tomato_Bacterial_spot |
| 7 | Tomato_Early_blight |
| 8 | Tomato_Late_blight |
| 9 | Tomato_Leaf_Mold |
| 10 | Tomato_Septoria_leaf_spot |
| 11 | Tomato_Spider_mites_Two_spotted_spider_mite |
| 12 | Tomato__Target_Spot |
| 13 | Tomato__Tomato_YellowLeaf__Curl_Virus |
| 14 | Tomato__Tomato_mosaic_virus |
| 15 | Tomato_healthy |

### 长尾分布

* **最大类**：`Tomato__Tomato_YellowLeaf__Curl_Virus`（3,208 张）
* **最小类**：`Potato___healthy`（152 张）
* **极差**：约 **21 倍** —— 因此引入 **Class Weights** 缓解不平衡。

---

## 五、环境依赖

### `requirements.txt`

```
torch==2.10.0+cu128
torchvision==0.25.0+cu128
torchaudio==2.10.0+cu128
matplotlib==3.10.0
scikit-learn==1.6.1
streamlit==1.57.0
opendatasets==0.1.22
onnx==1.20.1
onnx2tf==2.4.0
tensorflow==2.20.0
numpy==1.26.4
pillow==11.3.0
pandas==2.2.2
```

### 安装

建议使用 conda（与 `.vscode/settings.json` 一致）：

```bash
conda create -n plantdisease python=3.11 -y
conda activate plantdisease
pip install -r requirements.txt
```

### Kaggle 凭证

`download_data.py` 使用 `opendatasets` 下载，需要 `kaggle.json`：

1. 登录 Kaggle → Account → Create New API Token
2. 下载的 `kaggle.json` 放到项目根目录或 `~/.kaggle/`

### 设备检测

```bash
python cuda_acailable.py
```

输出 `使用设备: cuda:0` 表示 GPU 可用。

---

## 六、模型与训练配置

| 配置项 | 值 |
|---|---|
| 主干网络 | `torchvision.models.mobilenet_v3_small` |
| 迁移学习 | 加载 ImageNet 预训练权重，仅替换 `classifier[3]` 为 15 维全连接 |
| 输入尺寸 | 224 × 224 |
| Batch size | 32 |
| 学习率 | 0.001 |
| 优化器 | Adam |
| 学习率调度 | `StepLR(step_size=7, gamma=0.1)` |
| 损失函数 | `CrossEntropyLoss`（可选类别权重） |
| Epochs | 15 |
| 保存策略 | 仅保存验证集准确率最高的 checkpoint |

### 数据增强（仅训练集）

```python
RandomResizedCrop(224)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.3)
RandomRotation(20)
RandomAffine(degrees=0, shear=10)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
GaussianBlur(kernel_size=(5,5), sigma=(0.1, 2.0))
ToTensor()
Normalize(ImageNet mean/std)
```

验证/测试集仅做 `Resize(256) → CenterCrop(224) → Normalize`。

### 类别权重计算

```python
class_counts = [len(os.listdir('data/train/<class>')) for class in classes]
weights = 1. / class_counts
weights = weights / weights.sum()
```

将归一化倒数频率作为 `CrossEntropyLoss(weight=...)`。

---

## 七、运行流程（分步骤）

### 步骤 1：下载并划分数据

```bash
python download_data.py
```

* 默认只执行 `split_data()`（脚本中 `download_data()` 已注释）。
* 首次使用前请取消 `download_data.py` 末尾对 `download_data()` 的注释，下载完成后再注释回去。
* 产物：`./data/{train,val,test}/<class>/*.JPG`

### 步骤 2：主训练（完整版模型）

```bash
python train_model.py
```

* 启用 **全部增强 + 类别权重 + ImageNet 预训练**
* 产物：
  * `plant_disease_mobilenet.pth`（项目根目录，最佳验证准确率权重）
  * `classes/classes.txt`（写入类别名）

### 步骤 3：主训练曲线可视化

```bash
python train_model_plt_analysis.py
```

* 依赖 `train_model.py` 的 `history` 变量。
* 建议在 IDE/Jupyter 中先 `from train_model import train_model; history = train_model()` 后再调用绘图函数。
* 产物：`png/train_model_plt_analysis.png`

### 步骤 4：消融实验（共 4 次训练）

进入 `ablation_experiment/` 目录依次运行：

```bash
cd ablation_experiment
python baseline_ablation_train_model.py        # 仅预训练
python augmentation_ablation_train_model.py    # +增强
python class_weights_ablation_train_model.py   # +增强+类别权重
python no_pretrained_ablation_train_model.py   # 增强+权重，但不预训练
```

四个变体共用 `ablation_train_model.py::train_model_ablation(use_augmentation, use_class_weights, use_pretrained, model_save_path)`：

| 变体 | use_augmentation | use_class_weights | use_pretrained |
|---|:-:|:-:|:-:|
| baseline | ✗ | ✗ | ✓ |
| augmentation | ✓ | ✗ | ✓ |
| class_weights | ✓ | ✓ | ✓ |
| no_pretrained | ✓ | ✓ | ✗ |

* 产物：4 份 `plant_disease_mobilenet_*.pth`（默认存到当前目录，**手动移动**到 `pth/ablation_experiment/`）。

### 步骤 5：消融实验定量评估

```bash
python ablation_experiment/abliation_model_evaluation.py
```

* 加载 4 个 `.pth` 在 **测试集** 上评估
* 输出每个模型的 **整体准确率** + **逐类召回率**
* 重点关注少数类（脚本示例：`Potato___healthy`）

### 步骤 6：消融实验对比图

```bash
python ablation_experiment/ablation_plt_analysis.py
```

* 在同一张图内对比 4 个变体的 train/val accuracy 与 loss 曲线
* 产物：`png/ablation_history_comparison.png`

### 步骤 7：鲁棒性评估

对消融实验中表现最佳的模型（默认 `class_weights`）注入扰动：

```bash
cd robustness_evaluation
python gaussian_noise_test.py    # std ∈ {0.01, 0.02, 0.04, 0.06, 0.08, 0.1}
python gaussian_blur_test.py     # kernel ∈ {3, 5, 7, 9, 11}
python robustness_test_plt_analysis.py  # 绘图
```

* 评估函数：`evaluate_robustness_fundefine.py::evaluate_robustness(model_path, test_dir, class_names, intensities, mode)`
  * `mode='noise'`：在归一化后的 tensor 上加 `randn * std`
  * `mode='blur'`：用 `GaussianBlur(kernel_size, sigma=(0.1, 2.0))`
* 产物：`png/robustness_test_plt_analysis.png`

---

## 八、实验方案

### 8.1 消融实验（Ablation Study）

**目标**：验证高准确率（99.35%）的来源。

| 对比项 | 验证目标 |
|---|---|
| **Baseline** | 仅预训练，作为参考下界 |
| **+Augmentation** | 数据增强带来的泛化提升 |
| **+ClassWeights** | 类别权重对少数类（如 `Potato___healthy`）召回率的提升 |
| **No Pretrained** | 迁移学习 vs 从零训练的收敛速度差距 |

### 8.2 鲁棒性测试（Robustness Testing）

**目标**：模拟实际复杂环境下的识别稳定性。

| 干扰类型 | 模拟场景 | 强度范围 |
|---|---|---|
| 高斯噪声 | 拍摄噪点 | std ∈ {0.01 ~ 0.1} |
| 高斯模糊 | 对焦不准 | kernel ∈ {3, 5, 7, 9, 11} |

**产出**：干扰强度-准确率退化曲线图。

### 8.3 模型可解释性（计划中）

* **技术**：Grad-CAM
* **目标**：可视化模型在番茄/马铃薯叶片上对"病斑"区域的响应，证明决策的科学性。

---

## 九、部署与 Android App

### 9.1 模型转换

转换路径：`.pth` → `.onnx` → `.tflite`

依赖：`onnx==1.20.1`, `onnx2tf==2.4.0`, `tensorflow==2.20.0`（已在 `requirements.txt`）。

### 9.2 量化对比实验

| 精度 | 用途 |
|---|---|
| **FP32** | 原始精度参考 |
| **FP16** | 平衡体积与精度 |
| **INT8** | 极致推理速度（需量化校准） |

**产出指标**：模型文件体积 (MB)、Params (参数量)、FLOPs (计算量)。

### 9.3 Android App

* **核心功能**：
  1. **识别模式**：点击拍照识别（离线静态识别）
  2. **结果展示**：类别名称、置信度、对应的病害防治建议
* **集成方式**：通过 `TFLiteClassifier` 模块封装模型推理代码，集成入 Android Studio 项目。

---

## 十、性能评估指标

* **测试设备**：realme GT 2
* **量化指标**：
  * **推理延迟 (Latency)**：单张图片平均处理耗时
  * **资源占用**：CPU 占用率、内存占用峰值

---

## 十一、关键脚本对照表

| 脚本 | 作用 | 产物 |
|---|---|---|
| `cuda_acailable.py` | 检测 GPU 可用性 | 控制台输出 |
| `download_data.py` | 下载 + 数据集划分 | `./data/{train,val,test}/` |
| `train_model.py` | 主训练（完整模型） | `plant_disease_mobilenet.pth`, `classes/classes.txt` |
| `train_model_plt_analysis.py` | 主训练曲线绘制 | `png/train_model_plt_analysis.png` |
| `ablation_experiment/ablation_train_model.py` | 通用训练函数定义 | — |
| `ablation_experiment/*_ablation_train_model.py` | 4 个消融变体 | 4 份 `.pth` |
| `ablation_experiment/abliation_model_evaluation.py` | 测试集评估 | 控制台准确率/召回率 |
| `ablation_experiment/ablation_plt_analysis.py` | 消融对比图 | `png/ablation_history_comparison.png` |
| `robustness_evaluation/evaluate_robustness_fundefine.py` | 鲁棒性评估函数定义 | — |
| `robustness_evaluation/gaussian_noise_test.py` | 高斯噪声测试 | `noise_results` dict |
| `robustness_evaluation/gaussian_blur_test.py` | 高斯模糊测试 | `blur_results` dict |
| `robustness_evaluation/robustness_test_plt_analysis.py` | 鲁棒性曲线绘图 | `png/robustness_test_plt_analysis.png` |

---

## 十二、运行注意事项

1. **工作目录敏感**
   多数脚本使用相对路径（`./data`、`./pth/...`、`./classes/classes.txt`）。建议**统一在项目根目录运行**；进入子目录运行时需要把路径前缀改为 `../`。

2. **类别顺序**
   `classes/classes.txt` 由 `train_model.py` 写入，**顺序与 `ImageFolder` 扫描结果一致**。后续评估脚本读取该文件以保持一致，**请勿手动修改顺序**，否则评估结果会错位。

3. **消融脚本会立刻执行训练**
   `baseline_ablation_train_model.py` 等子脚本在顶层直接调用了 `train_model_ablation(...)`。若在 `ablation_plt_analysis.py` 中 `import` 它们，会触发 4 次完整训练。**建议先单独跑完 4 个训练脚本，再单独跑对比绘图**。

4. **权重文件位置**
   * `train_model.py` 默认保存到项目根目录。
   * 消融实验脚本默认保存到运行时 CWD（多为 `ablation_experiment/`）。
   * 评估脚本读取的是 `pth/ablation_experiment/`。
   * 训练完成后，**请手动将 `.pth` 移动到 `pth/ablation_experiment/`**。

5. **Kaggle 数据下载**
   `download_data.py` 中 `download_data()` 默认被注释。若数据已在 `plantdisease/PlantVillage/` 下，可直接跑 `split_data()`。

---

## 十三、许可与致谢

* **数据集**：[PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
* **主干网络**：`torchvision.models.mobilenet_v3_small`
