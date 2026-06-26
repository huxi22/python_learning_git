import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

class UWB_CRNN(nn.Module):
    def __init__(self, cir_len = 640, num_bins = 200):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d( 1, 16, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2)
        )

        self.projection = nn.Sequential(
            nn.Linear(32*(cir_len//4), 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_bins), nn.Sigmoid()
        )


    def forward(self, x):
        batch, seq, cir = x.size()
        x = x.view(batch*seq, 1, cir)

        feat = self.cnn(x)
        feat = feat.view(feat.size(0), -1)

        feat = self.projection(feat)
        lstm_in = feat.view(batch, seq, -1)

        lstm_out, _ = self.lstm(lstm_in)
        out  = self.decoder(lstm_out)

        return out


def main_pipeline(fullDataSet):
    # ---------- 模型配置 ---------- #
    # 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 划分数据集
    total_size = len(fullDataSet)
    train_size = int(total_size * 0.7)
    val_size   = int(total_size * 0.1)
    test_size  = total_size - train_size - val_size
    train_set, val_set, test_set = random_split(
        fullDataSet, 
        [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=2, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=2, shuffle=False)

    # 模型和优化器初始化
    model = UWB_CRNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    # ---------- 训练流程 ---------- #
    print("\n=== 开始训练/验证 ===")

    epochs = 100
    best_val_loss = float("inf")

    for epoch in range(epochs):

        # ----- 训练集上训练模型 ----- #
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            # 把输入数据放到device上
            inputs, targets = inputs.to(device), targets.to(device).float()

            # 模型梯度置为0
            optimizer.zero_grad()

            # 计算loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 梯度反向传播
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()

        ave_train_loss = train_loss/len(train_loader)
        

        # ----- 验证集上验证模型训练结果 ----- #
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                # 把输入数据放到device上
                inputs, targets = inputs.to(device), targets.to(device).float()

                # 计算loss
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

        ave_val_loss = val_loss/len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train loss: {ave_train_loss:.4f}, Val loss: {ave_val_loss:.4f}")


        # ----- 保存验证集上最好的模型 ----- #
        if best_val_loss > ave_val_loss:
            best_val_loss = ave_val_loss
            torch.save(model.state_dict(), "best_uwb_crnn.pth")

    print("模型训练完成，可以保存用于部署了。")


    # ---------- 测试流程 ---------- #
    print("\n=== 开始测试 ===")

    # 导入最佳模型参数
    model.load_state_dict(torch.load("best_uwb_crnn"))
    model.eval()

    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            # 把输入数据放到device上
            inputs, targets = inputs.to(device), targets.to(device).float()

            # 计算loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
        
    ave_test_loss = test_loss/len(test_loader)
    print(f"Test loss: {ave_test_loss:.4f}")


if __name__ == '__main__':
    main_pipeline(fullDataSet)

















