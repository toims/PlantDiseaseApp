import shutil
import random
import os
import opendatasets as od
from pathlib import Path

DATASET_URL = 'https://www.kaggle.com/datasets/emmarex/plantdisease'
RAW_DATA_DIR = './plantdisease/PlantVillage/PlantVillage'
OUTPUT_DIR = './data'

# 设置划分比例 7:1.5:1.5
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def download_data():
  if os.path.exists(RAW_DATA_DIR):
    print(f"{RAW_DATA_DIR}已存在，跳过下载")
    return
  print("正在下载数据集，可能要数分钟...")
  try:
    od.download(DATASET_URL)
    print("下载数据集成功")
  except Exception as e:
    print(e)


def split_data():
  if os.path.exists(OUTPUT_DIR):
    print(f"{OUTPUT_DIR}已存在")
    user_input = input("是否删除旧数据并重新划分？(y/n): ")
    if user_input.lower() == 'y':
      shutil.rmtree(OUTPUT_DIR)
    else:
      print("已取消操作")
      return

  # 定义三个目录路径
  train_dir = os.path.join(OUTPUT_DIR, 'train')
  val_dir = os.path.join(OUTPUT_DIR, 'val')
  test_dir = os.path.join(OUTPUT_DIR, 'test')

  for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

  source_root = Path(RAW_DATA_DIR)
  if not source_root.exists():
    source_root = Path('./plantdisease/PlantVillage')
    if not source_root.exists():
      print(f"ERROR : 找不到原始数据路径{RAW_DATA_DIR}.")
      return

  classes = [d for d in os.listdir(source_root) if os.path.isdir(source_root / d)]
  print(f"找到 {len(classes)} 个类别，开始按 7:1.5:1.5 的比例划分...")

  total_images = 0

  for class_name in classes:
    class_path = source_root / class_name
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    random.shuffle(images)

    # 计算切分点
    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]

    # 创建类别子目录
    for d in [train_dir, val_dir, test_dir]:
      os.makedirs(os.path.join(d, class_name), exist_ok=True)

    # 拷贝文件
    for img in train_images:
      shutil.copy2(class_path / img, os.path.join(train_dir, class_name, img))
    for img in val_images:
      shutil.copy2(class_path / img, os.path.join(val_dir, class_name, img))
    for img in test_images:
      shutil.copy2(class_path / img, os.path.join(test_dir, class_name, img))

    print(f"类别：{class_name:45} | 训练:{len(train_images)} | 验证:{len(val_images)} | 测试:{len(test_images)}")
    total_images += n_total

  print("-"*30)
  print(f"处理完成! 共处理 {total_images} 张照片")
  print(f"位置: {train_dir}, {val_dir}, {test_dir}")

if __name__ == "__main__":
  #download_data()
  print("当前模块已经执行完成！！！")
  print("还有split_data()函数未执行 [数据集分割]")
  split_data()
  