

# 执行消融试验

## 第一项消融试验

```markdown
the train_model_ablation function is defined, run the first ablation experiment: the baseline model. This involves setting use_augmentation=False, use_class_weights=False, and use_pretrained=True, and specifying a unique save path for this model. This will train the model and store its performance history.
train_model_ablation 函数已经定义，运行第一个消融实验：基线模型。这包括设置 use_augmentation=False，use_class_weights=False，use_pretrained=True，并为该模型指定一个唯一的保存路径。这将训练模型并存储其性能历史。
```

### 代码块

```python
baseline_history = train_model_ablation(
    use_augmentation=False,
    use_class_weights=False,
    use_pretrained=True,
    model_save_path='plant_disease_mobilenet_baseline.pth'
)
print("Baseline model training complete. History stored in `baseline_history`.")
```

### 基线模型logs

```markdown
開始訓練 (Augmentation: False, Class Weights: False, Pretrained: True)... 類別數: 15
Epoch 1/15
----------
train Loss: 0.2718 Acc: 0.9117
val Loss: 0.2894 Acc: 0.9087
Epoch 2/15
----------
train Loss: 0.0969 Acc: 0.9695
val Loss: 0.1268 Acc: 0.9615
Epoch 3/15
----------
train Loss: 0.0849 Acc: 0.9735
val Loss: 0.1119 Acc: 0.9624
Epoch 4/15
----------
train Loss: 0.0465 Acc: 0.9848
val Loss: 0.0489 Acc: 0.9822
Epoch 5/15
----------
train Loss: 0.0513 Acc: 0.9841
val Loss: 0.1293 Acc: 0.9615
Epoch 6/15
----------
train Loss: 0.0424 Acc: 0.9856
val Loss: 0.3686 Acc: 0.9145
Epoch 7/15
----------
train Loss: 0.0465 Acc: 0.9854
val Loss: 0.6248 Acc: 0.8482
Epoch 8/15
----------
train Loss: 0.0137 Acc: 0.9954
val Loss: 0.0242 Acc: 0.9919
Epoch 9/15
----------
train Loss: 0.0057 Acc: 0.9985
val Loss: 0.0196 Acc: 0.9942
Epoch 10/15
----------
train Loss: 0.0027 Acc: 0.9992
val Loss: 0.0169 Acc: 0.9951
Epoch 11/15
----------
train Loss: 0.0025 Acc: 0.9995
val Loss: 0.0183 Acc: 0.9948
Epoch 12/15
----------
train Loss: 0.0033 Acc: 0.9990
val Loss: 0.0176 Acc: 0.9951
Epoch 13/15
----------
train Loss: 0.0021 Acc: 0.9994
val Loss: 0.0198 Acc: 0.9945
Epoch 14/15
----------
train Loss: 0.0020 Acc: 0.9994
val Loss: 0.0134 Acc: 0.9968
Epoch 15/15
----------
train Loss: 0.0017 Acc: 0.9996
val Loss: 0.0135 Acc: 0.9968
訓練完成! 最佳驗證準確率: 0.9968
Baseline model training complete. History stored in `baseline_history`.
```

## 第二项消融试验

```markdown
the second ablation experiment, call the train_model_ablation function with data augmentation enabled, class weights disabled, and pretrained weights enabled, saving the model to a new path and storing the training history.
行第二个消融实验，我将调用train_model_ablation函数，启用数据增强，禁用类别权重，并启用预训练权重，将模型保存到新的路径，并存储训练历史。
```

### 代码块

```python
augmentation_history = train_model_ablation(
    use_augmentation=True,
    use_class_weights=False,
    use_pretrained=True,
    model_save_path='plant_disease_mobilenet_augmentation.pth'
)
print("Model with augmentation training complete. History stored in `augmentation_history`.")
```

### 加入数据增强模型logs

```markdown
開始訓練 (Augmentation: True, Class Weights: False, Pretrained: True)... 類別數: 15
Epoch 1/15
----------
train Loss: 0.5635 Acc: 0.8184
val Loss: 0.2554 Acc: 0.9194
Epoch 2/15
----------
train Loss: 0.2752 Acc: 0.9076
val Loss: 0.2107 Acc: 0.9346
Epoch 3/15
----------
train Loss: 0.2402 Acc: 0.9200
val Loss: 0.2144 Acc: 0.9304
Epoch 4/15
----------
train Loss: 0.2036 Acc: 0.9335
val Loss: 0.1252 Acc: 0.9527
Epoch 5/15
----------
train Loss: 0.1914 Acc: 0.9371
val Loss: 0.1932 Acc: 0.9421
Epoch 6/15
----------
train Loss: 0.1850 Acc: 0.9383
val Loss: 0.1294 Acc: 0.9654
Epoch 7/15
----------
train Loss: 0.1544 Acc: 0.9515
val Loss: 0.1018 Acc: 0.9621
Epoch 8/15
----------
train Loss: 0.0966 Acc: 0.9675
val Loss: 0.0258 Acc: 0.9919
Epoch 9/15
----------
train Loss: 0.0759 Acc: 0.9744
val Loss: 0.0189 Acc: 0.9932
Epoch 10/15
----------
train Loss: 0.0721 Acc: 0.9756
val Loss: 0.0155 Acc: 0.9948
Epoch 11/15
----------
train Loss: 0.0657 Acc: 0.9784
val Loss: 0.0164 Acc: 0.9948
Epoch 12/15
----------
train Loss: 0.0609 Acc: 0.9789
val Loss: 0.0123 Acc: 0.9958
Epoch 13/15
----------
train Loss: 0.0643 Acc: 0.9792
val Loss: 0.0146 Acc: 0.9958
Epoch 14/15
----------
train Loss: 0.0563 Acc: 0.9816
val Loss: 0.0143 Acc: 0.9958
Epoch 15/15
----------
train Loss: 0.0557 Acc: 0.9821
val Loss: 0.0121 Acc: 0.9964
訓練完成! 最佳驗證準確率: 0.9964
Model with augmentation training complete. History stored in `augmentation_history`.
```

## 第三项消融试验

```markdown
the third ablation experiment,call the train_model_ablation function with data augmentation, class weights, and pretrained weights enabled, saving the model to a new path and storing the training history.
第三个消融实验，我将调用train_model_ablation函数，并启用数据增强、类别权重和预训练权重，将模型保存到新路径并存储训练历史。
```

### 代码块

```python
class_weights_history = train_model_ablation(
    use_augmentation=True,
    use_class_weights=True,
    use_pretrained=True,
    model_save_path='plant_disease_mobilenet_class_weights.pth'
)
print("Model with augmentation and class weights training complete. History stored in `class_weights_history`.")
```

### 加入类别权重logs

```markdown
開始訓練 (Augmentation: True, Class Weights: True, Pretrained: True)... 類別數: 15
Epoch 1/15
----------
train Loss: 0.6280 Acc: 0.8012
val Loss: 0.3424 Acc: 0.8754
Epoch 2/15
----------
train Loss: 0.3224 Acc: 0.8909
val Loss: 0.3584 Acc: 0.8906
Epoch 3/15
----------
train Loss: 0.2945 Acc: 0.9044
val Loss: 0.3516 Acc: 0.9048
Epoch 4/15
----------
train Loss: 0.2445 Acc: 0.9175
val Loss: 0.2631 Acc: 0.8977
Epoch 5/15
----------
train Loss: 0.2127 Acc: 0.9264
val Loss: 0.0858 Acc: 0.9699
Epoch 6/15
----------
train Loss: 0.2210 Acc: 0.9286
val Loss: 0.2214 Acc: 0.9230
Epoch 7/15
----------
train Loss: 0.2041 Acc: 0.9309
val Loss: 0.1072 Acc: 0.9680
Epoch 8/15
----------
train Loss: 0.1185 Acc: 0.9598
val Loss: 0.0336 Acc: 0.9871
Epoch 9/15
----------
train Loss: 0.1013 Acc: 0.9652
val Loss: 0.0293 Acc: 0.9909
Epoch 10/15
----------
train Loss: 0.0897 Acc: 0.9677
val Loss: 0.0285 Acc: 0.9896
Epoch 11/15
----------
train Loss: 0.0854 Acc: 0.9699
val Loss: 0.0293 Acc: 0.9896
Epoch 12/15
----------
train Loss: 0.0773 Acc: 0.9720
val Loss: 0.0191 Acc: 0.9916
Epoch 13/15
----------
train Loss: 0.0692 Acc: 0.9752
val Loss: 0.0214 Acc: 0.9922
Epoch 14/15
----------
train Loss: 0.0705 Acc: 0.9737
val Loss: 0.0217 Acc: 0.9938
Epoch 15/15
----------
train Loss: 0.0702 Acc: 0.9748
val Loss: 0.0189 Acc: 0.9945
訓練完成! 最佳驗證準確率: 0.9945
Model with augmentation and class weights training complete. History stored in `class_weights_history`.
```

## 第四项消融试验

```markdown
execute the fourth ablation experiment, call the train_model_ablation function with data augmentation and class weights enabled, but with pretrained weights disabled. The training history will be stored in a dedicated variable, and the model will be saved to a new path.
执行第四次消融实验，调用train_model_ablation函数，并启用数据增强和类别权重，但禁用预训练权重。训练历史将存储在一个专用变量中，模型将保存到新的路径
```

### 代码块

```python
no_pretrained_history = train_model_ablation(
    use_augmentation=True,
    use_class_weights=True,
    use_pretrained=False,
    model_save_path='plant_disease_mobilenet_no_pretrained.pth'
)
print("Model without pretrained weights training complete. History stored in `no_pretrained_history`.")
```

### 无迁移学习logs

```markdown
開始訓練 (Augmentation: True, Class Weights: True, Pretrained: False)... 類別數: 15
Epoch 1/15
----------
train Loss: 2.1851 Acc: 0.2963
val Loss: 2.7013 Acc: 0.2273
Epoch 2/15
----------
train Loss: 1.5066 Acc: 0.5082
val Loss: 1.0620 Acc: 0.6556
Epoch 3/15
----------
train Loss: 1.1283 Acc: 0.6320
val Loss: 0.9798 Acc: 0.6591
Epoch 4/15
----------
train Loss: 0.9073 Acc: 0.6992
val Loss: 0.5373 Acc: 0.8051
Epoch 5/15
----------
train Loss: 0.7471 Acc: 0.7514
val Loss: 0.4989 Acc: 0.8239
Epoch 6/15
----------
train Loss: 0.6523 Acc: 0.7843
val Loss: 0.5525 Acc: 0.8019
Epoch 7/15
----------
train Loss: 0.6092 Acc: 0.8022
val Loss: 0.5848 Acc: 0.8106
Epoch 8/15
----------
train Loss: 0.4343 Acc: 0.8553
val Loss: 0.1881 Acc: 0.9304
Epoch 9/15
----------
train Loss: 0.3790 Acc: 0.8751
val Loss: 0.1766 Acc: 0.9353
Epoch 10/15
----------
train Loss: 0.3659 Acc: 0.8769
val Loss: 0.1782 Acc: 0.9356
Epoch 11/15
----------
train Loss: 0.3379 Acc: 0.8886
val Loss: 0.1497 Acc: 0.9479
Epoch 12/15
----------
train Loss: 0.3506 Acc: 0.8839
val Loss: 0.1594 Acc: 0.9391
Epoch 13/15
----------
train Loss: 0.3292 Acc: 0.8906
val Loss: 0.1749 Acc: 0.9388
Epoch 14/15
----------
train Loss: 0.3309 Acc: 0.8899
val Loss: 0.1578 Acc: 0.9446
Epoch 15/15
----------
train Loss: 0.3183 Acc: 0.8968
val Loss: 0.1321 Acc: 0.9537
訓練完成! 最佳驗證準確率: 0.9537
Model without pretrained weights training complete. History stored in `no_pretrained_history`.
```

# 消融试验图像分析

```markdown
the plot_ablation_history_comparison function is defined, call it to compare the training histories of the four ablation experiments that were just conducted. This will provide a visual analysis of how each component (data augmentation, class weights, pretrained weights) impacts the model's performance.
定义了 plot_ablation_history_comparison 函数，调用它来比较刚刚进行的四个消融实验的训练历史。这将提供一种直观的分析方式，展示每个组件（数据增强、类别权重、预训练权重）对模型性能的影响。
```

```python
histories_to_compare = [
    baseline_history,
    augmentation_history,
    class_weights_history,
    no_pretrained_history
]

labels_for_comparison = [
    'Baseline (No Aug, No CW, Pretrained)',
    '+ Augmentation (No CW, Pretrained)',
    '+ Class Weights (Aug, Pretrained)',
    'No Pretrained (Aug, CW)'
]

plot_ablation_history_comparison(histories_to_compare, labels_for_comparison)
print("Generated comparison plots for ablation study.")
```

![](..\png\ablation_history_comparison.png)

# 消融试验定量分析

```markdown
Using device: cuda:0
Evaluated model: plant_disease_mobilenet_baseline.pth
Overall Test Accuracy: 0.9955
Per-class Recall:
  Pepper__bell___Bacterial_spot: 0.9934
  Pepper__bell___healthy: 1.0000
  Potato___Early_blight: 1.0000
  Potato___Late_blight: 1.0000
  Potato___healthy: 1.0000
  Tomato_Bacterial_spot: 1.0000
  Tomato_Early_blight: 0.9733
  Tomato_Late_blight: 0.9930
  Tomato_Leaf_Mold: 1.0000
  Tomato_Septoria_leaf_spot: 0.9813
  Tomato_Spider_mites_Two_spotted_spider_mite: 1.0000
  Tomato__Target_Spot: 0.9906
  Tomato__Tomato_YellowLeaf__Curl_Virus: 1.0000
  Tomato__Tomato_mosaic_virus: 1.0000
  Tomato_healthy: 1.0000
Evaluated model: plant_disease_mobilenet_augmentation.pth
Overall Test Accuracy: 0.9965
Per-class Recall:
  Pepper__bell___Bacterial_spot: 1.0000
  Pepper__bell___healthy: 1.0000
  Potato___Early_blight: 1.0000
  Potato___Late_blight: 1.0000
  Potato___healthy: 1.0000
  Tomato_Bacterial_spot: 1.0000
  Tomato_Early_blight: 0.9867
  Tomato_Late_blight: 0.9930
  Tomato_Leaf_Mold: 0.9931
  Tomato_Septoria_leaf_spot: 0.9963
  Tomato_Spider_mites_Two_spotted_spider_mite: 0.9841
  Tomato__Target_Spot: 0.9953
  Tomato__Tomato_YellowLeaf__Curl_Virus: 1.0000
  Tomato__Tomato_mosaic_virus: 1.0000
  Tomato_healthy: 1.0000
Evaluated model: plant_disease_mobilenet_class_weights.pth
Overall Test Accuracy: 0.9968
Per-class Recall:
  Pepper__bell___Bacterial_spot: 1.0000
  Pepper__bell___healthy: 1.0000
  Potato___Early_blight: 1.0000
  Potato___Late_blight: 1.0000
  Potato___healthy: 0.9583
  Tomato_Bacterial_spot: 1.0000
  Tomato_Early_blight: 0.9800
  Tomato_Late_blight: 0.9965
  Tomato_Leaf_Mold: 1.0000
  Tomato_Septoria_leaf_spot: 1.0000
  Tomato_Spider_mites_Two_spotted_spider_mite: 0.9960
  Tomato__Target_Spot: 0.9906
  Tomato__Tomato_YellowLeaf__Curl_Virus: 0.9959
  Tomato__Tomato_mosaic_virus: 1.0000
  Tomato_healthy: 1.0000
Evaluated model: plant_disease_mobilenet_no_pretrained.pth
Overall Test Accuracy: 0.9492
Per-class Recall:
  Pepper__bell___Bacterial_spot: 0.9669
  Pepper__bell___healthy: 0.9686
  Potato___Early_blight: 0.9867
  Potato___Late_blight: 0.9400
  Potato___healthy: 1.0000
  Tomato_Bacterial_spot: 0.9469
  Tomato_Early_blight: 0.9000
  Tomato_Late_blight: 0.8641
  Tomato_Leaf_Mold: 1.0000
  Tomato_Septoria_leaf_spot: 0.9588
  Tomato_Spider_mites_Two_spotted_spider_mite: 0.9563
  Tomato__Target_Spot: 0.9151
  Tomato__Tomato_YellowLeaf__Curl_Virus: 0.9585
  Tomato__Tomato_mosaic_virus: 1.0000
  Tomato_healthy: 0.9833

--- Ablation Study Quantitative Results ---
Baseline Model: Overall Accuracy = 0.9955
Augmentation Model: Overall Accuracy = 0.9965
Class Weights Model: Overall Accuracy = 0.9968
No Pretrained Model: Overall Accuracy = 0.9492

Recall for 'Potato___healthy':
  Baseline: 1.0000
  Augmentation: 1.0000
  Class Weights: 0.9583
  No Pretrained: 1.0000
```

# 分析少数类别召回率：'Potato___healthy'

```markdown
目的：评估数据增强和类别权重对少数类别召回率的影响，并探讨其与整体准确度的关系。

'Potato___healthy' 類別召回率比較：
基準模型（Baseline）：
整体准确度: 0.9955
Potato___healthy 召回率: 1.0000
+数据增强模型（+Augmentation）：
整体准确度: 0.9965
Potato___healthy 召回率: 1.0000
+类别权重模型（+Class Weights）：
整体准确度: 0.9968
Potato___healthy 召回率: 0.9583
无迁移学习模型（No Pretrained）：
整体准确度: 0.9492
Potato___healthy 召回率: 1.0000

分析与总结：
数据增强的影响：从基准模型到加入数据增强的模型，Potato___healthy 的召回率保持在 1.0000 不变，而整体准确度略有提升(0.9955 -> 0.9965)。这表明数据增强在不损害少数类别识别能力的同时，有助于提高模型的整体泛化能力。

类别权重的影响：有趣的是，在引入类别权重后，尽管整体准确率达到了最高 (0.9968)，Potato___healthy 的召回率反而从 1.0000 下降到 0.9583。这可能表明在某些情况下，当模型已经通过其他机制（如预训练模型）对所有类别学习得较好时，类别权重可能会导致模型对某些原本表现极佳的少数类别过度“补偿”，反而使其性能略有下降。这也可能是由于Potato___healthy 虽然属于少数类别，但其特征可能相对清晰，模型在基准情况下已经能够完美识别，而类别权重调整了损失函数的焦点，间接影响了其表现。

迁移学习的影响：没有迁移学习的模型在整体准确度 (0.9492) 上远低于其他使用预训练权重的模型，这凸显了迁移学习的巨大优势。尽管如此，该模型的 Potato___healthy 召回率仍然保持在 1.0000。这再次说明 Potato___healthy 类别可能相对容易识别，即使在整体性能较差的模型中也能有良好表现。

整体准确度与少数类别召回率的关系：

高整体准确度并不总是意味着所有类别都表现完美，尤其是对于少数类别。例如，类别权重模型在整体准确度最高的情况下，Potato___healthy 的召回率却略有下降。
在某些情况下，少数类别的召回率可能已经很高（如 Potato___healthy），此时数据增强或类别权重的引入，主要影响的是其他类别的平衡或整体性能。
结论：对于 Potato___healthy 这一少数类别，其在基准模型和数据增强模型中已达到完美召回率。引入类别权重后，虽然整体准确度略有提高，但该特定少数类别的召回率反而略有下降，这提示我们在类别分布非常不平衡且少数类别已经表现良好的情况下，引入类别权重需要谨慎评估其对每个类别的具体影响。迁移学习对整体模型性能至关重要，即使没有迁移学习，Potato___healthy 也能被很好地识别。
```

