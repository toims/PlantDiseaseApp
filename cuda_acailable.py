import torch

# 自動檢測設備
def cuda_available():
  DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"使用设备: {DEVICE}")

if __name__ == '__main__':
  cuda_available()
