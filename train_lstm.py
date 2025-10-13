# train_lstm.py  —— mini-batch + memmap 版本
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lstm_model import LSTM_RUL

# ---------- 1) 数据集（按索引逐条读取，避免一次性占满内存） ----------
class NpySequenceDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, x_dtype=np.float32):
        # 不把全部读入内存，mmap 只在访问时加载
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        assert len(self.x) == len(self.y), "X/Y 数量不一致"
        self.x_dtype = x_dtype

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 只把当前索引的数据拷到内存，并转成 tensor
        x = np.asarray(self.x[idx], dtype=self.x_dtype)  # [T, F]
        y = np.asarray(self.y[idx], dtype=np.float32)    # 标量
        x = torch.from_numpy(x)                          # float32 tensor
        y = torch.from_numpy(y).view(1)                  # [1]
        return x, y

# ---------- 2) 路径 ----------
BASE = Path(__file__).resolve().parent
X_PATH = BASE / "data" / "processed" / "X_sequences.npy"
Y_PATH = BASE / "data" / "processed" / "Y_sequences.npy"

# ---------- 3) DataLoader（小批次） ----------
batch_size = 64           # 如仍 OOM，改成 32 / 16
num_workers = 0           # Windows 先用 0，避免多进程开销
pin_memory = False        # CPU 训练设 False；若用 GPU 可改 True

ds = NpySequenceDataset(X_PATH, Y_PATH, x_dtype=np.float32)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

# ---------- 4) 设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- 5) 模型 ----------
# 从一条样本推断输入维度
sample_x, _ = ds[0]
input_size = sample_x.shape[1]      # 每个时间步的特征数 F
model = LSTM_RUL(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------- 6) 训练循环 ---------- higher is better, if you dont' have a nvidia gpu, change epochs to 10
epochs = 30
model.train()
for ep in range(1, epochs + 1):
    running = 0.0
    n = 0
    for xb, yb in dl:
        # xb: [B, T, F]  yb: [B, 1]
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        # 可选：梯度裁剪，稳定训练
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running += loss.item() * xb.size(0)
        n += xb.size(0)

    print(f"Epoch {ep}/{epochs}  Loss: {running / n:.4f}")

# ---------- 7) 保存 ----------
torch.save(model.state_dict(), BASE / "lstm_model.pt")
print("Saved to lstm_model.pt")
