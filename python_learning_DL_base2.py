import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 构造一个假的 UWB CIR 数据集
# =========================

class FakeCIRDataset(Dataset):
    """
    生成模拟 CIR 数据。
    每个样本是一条长度为 cir_len 的一维 CIR。
    Autoencoder 的训练目标是重构输入本身，所以返回 x, x。
    """

    def __init__(self, num_samples=1000, cir_len=640):
        self.num_samples = num_samples
        self.cir_len = cir_len

        self.data = []

        for _ in range(num_samples):
            cir = self.generate_fake_cir(cir_len)
            self.data.append(cir)

        self.data = np.array(self.data, dtype=np.float32)

        # 简单归一化：整体标准化
        mean = self.data.mean()
        std = self.data.std() + 1e-8
        self.data = (self.data - mean) / std

    def generate_fake_cir(self, cir_len):
        """
        生成一条假的 CIR：
        - 随机噪声
        - 若干个高斯峰，模拟多径 / 目标反射
        """
        x = np.arange(cir_len)

        cir = 0.05 * np.random.randn(cir_len)

        # 模拟 2~5 个多径峰
        num_peaks = np.random.randint(2, 6)

        for _ in range(num_peaks):
            center = np.random.randint(50, cir_len - 50)
            width = np.random.uniform(2, 8)
            amp = np.random.uniform(0.5, 2.0)

            peak = amp * np.exp(-0.5 * ((x - center) / width) ** 2)
            cir += peak

        return cir

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cir = self.data[idx]

        # Conv1d 输入需要 [channel, length]
        cir = torch.tensor(cir).unsqueeze(0)  # [1, 640]

        # Autoencoder：输入和标签相同
        return cir, cir


# =========================
# 2. 定义 1D-CNN Autoencoder
# =========================

class CNNAutoencoder1D(nn.Module):
    def __init__(self, cir_len=640, latent_dim=128):
        super().__init__()

        self.cir_len = cir_len
        self.latent_dim = latent_dim

        # Encoder: [B, 1, 640] -> [B, 32, 160]
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 640 -> 320

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)   # 320 -> 160
        )

        # CNN 输出展平后是 32 * (cir_len // 4)
        self.flatten_dim = 32 * (cir_len // 4)

        # 压缩成 latent vector
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim),
            nn.ReLU()
        )

        # 从 latent vector 恢复到 CNN 特征图尺寸
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim),
            nn.ReLU()
        )

        # Decoder: [B, 32, 160] -> [B, 1, 640]
        self.decoder_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 160 -> 320
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 320 -> 640
            nn.Conv1d(16, 1, kernel_size=9, padding=4)
        )

    def encode(self, x):
        """
        x: [B, 1, 640]
        z: [B, latent_dim]
        """
        feat = self.encoder_cnn(x)              # [B, 32, 160]
        feat = feat.view(feat.size(0), -1)      # [B, 5120]
        z = self.encoder_fc(feat)               # [B, latent_dim]
        return z

    def decode(self, z):
        """
        z: [B, latent_dim]
        x_hat: [B, 1, 640]
        """
        feat = self.decoder_fc(z)               # [B, 5120]
        feat = feat.view(z.size(0), 32, self.cir_len // 4)  # [B, 32, 160]
        x_hat = self.decoder_cnn(feat)          # [B, 1, 640]
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# =========================
# 3. 训练函数
# =========================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)    # [B, 1, 640]
        targets = targets.to(device)  # [B, 1, 640]

        optimizer.zero_grad()

        outputs, z = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


# =========================
# 4. 验证 / 测试函数
# =========================

def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, z = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


# =========================
# 5. 可视化重构结果
# =========================

def plot_reconstruction(model, data_loader, device, sample_idx=0):
    model.eval()

    inputs, targets = next(iter(data_loader))
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs, z = model(inputs)

    original = inputs[sample_idx, 0].cpu().numpy()
    reconstructed = outputs[sample_idx, 0].cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(original, label="Original CIR")
    plt.plot(reconstructed, label="Reconstructed CIR", linestyle="--")
    plt.xlabel("CIR sample index")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.title("CNN Autoencoder Reconstruction")
    plt.tight_layout()
    plt.show()


# =========================
# 6. 主流程
# =========================

def main():
    # --- A. 设备配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- B. 数据准备 ---
    full_dataset = FakeCIRDataset(num_samples=1000, cir_len=640)

    train_size = 800
    val_size = 100
    test_size = 100

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # --- C. 模型、损失函数、优化器 ---
    model = CNNAutoencoder1D(cir_len=640, latent_dim=128).to(device)

    # Autoencoder 是重构任务，常用 MSELoss
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- D. 训练循环 ---
    num_epochs = 50
    best_val_loss = float("inf")

    print("\n=== Start Training ===")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # 保存验证集表现最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_cnn_autoencoder.pth")

    # --- E. 测试 ---
    print("\n=== Start Testing ===")

    model.load_state_dict(
    torch.load(
        "best_cnn_autoencoder.pth",
        map_location=device,
        weights_only=True
    )
)

    test_loss = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(f"Test Loss: {test_loss:.6f}")

    # --- F. 可视化重构效果 ---
    plot_reconstruction(model, test_loader, device)


if __name__ == "__main__":
    main()