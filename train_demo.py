import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time

# ==========================================
# 1. 模型定义 (复用之前优化后的结构)
# 这里的网络设计是PyTorch 代码的黄金标准（Best Practice）：
# 将网络按功能块（Block）打包在 __init__ 里，
# 然后在 forward 里赋予它们具有物理意义的独立变量名，
# ==========================================
class UWB_CRNN_Optimized(nn.Module):
    def __init__(self, cir_len=640, num_targets_bins=200):

        super(UWB_CRNN_Optimized, self).__init__()  # <-- 固定写法，初始化父类。 Python 3 引入了无参数的 super() 调用。当你直接写 super() 时，Python 解释器会施展“编译器魔法”。
                                                    # 它会自动通过上下文（调用栈）检测出当前代码所处的类名是什么，当前的实例对象 self 是哪个，然后在底层默默地帮你把这两个参数填上去。
                                                    # 优点：代码极其简洁；完全解耦了类名，就算你以后给类改名，或者到处复制粘贴代码，也绝对不会出错。
        # CNN: 特征提取
        self.cnn = nn.Sequential(                   # 可以视为级联系统，在底层它是一个容器（Container）。
                                                    # 当你把一堆网络层按顺序塞进 nn.Sequential 时，它会自动在内部帮你把前一层的输出连到下一层的输入上。
                                                    # nn.Sequential 的致命的限制：它只支持“单线直通”的网络。内部的每一层，必须只有一个输入，且只有一个输出。
                                                    # 后续当你准备引入 U-Net 结构的跳跃连接以保留细节信息时 ，nn.Sequential 就无能为力了。
                                                    # 因为跳跃连接意味着你需要把流水线第一道工序的产品，直接“空投”到最后一道工序去组合，这打破了单线直通的规则。
                                                    # 遇到那种情况该怎么办？ 很简单，把需要跳跃的部分拆开。
            nn.Conv1d(1, 16, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2)
        )
        # 投影层: 降维 (5120 -> 128)
        self.projection = nn.Sequential(
            nn.Linear(32 * (cir_len // 4), 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # LSTM: 时序记忆 (Input 128 -> Hidden 256)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        # MLP: 解码成概率图 (Hidden 256 -> Output 200)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_targets_bins), nn.Sigmoid() # 输出 0-1 的概率
        )

    def forward(self, x):
        batch, seq, cir = x.size()
        # 1. Fold: (B*T, 1, L)
        x = x.view(batch * seq, 1, cir)
        # 2. CNN
        feat = self.cnn(x)
        feat = feat.view(feat.size(0), -1) # Flatten
        # 3. Projection
        feat = self.projection(feat)
        # 4. Unfold: (B, T, 128)
        lstm_in = feat.view(batch, seq, -1)
        # 5. LSTM
        lstm_out, _ = self.lstm(lstm_in)
        # 6. MLP
        out = self.decoder(lstm_out)
        return out

# ==========================================
# 2. 数据集模拟 (Dataset)
# ==========================================
class FakeUWBDataset(Dataset):
    def __init__(self, total_samples=50):
        # 模拟 50 次实验，每次 100 帧，每帧 640 点
        self.data = torch.randn(total_samples, 100, 640)
        # 模拟真值：输出是 200 个距离 Bin 的概率 (0或1)
        self.label = torch.rand(total_samples, 100, 200) # 暂用随机数代替

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# ==========================================
# 3. 核心：训练与测试逻辑
# ==========================================
def main_pipeline():
    # --- A. 硬件配置 ---
    # 有显卡用显卡，没显卡用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # --- B. 数据准备 ---
    full_dataset = FakeUWBDataset(total_samples=50)
    
    # 划分训练集(40组) 和 测试集(10组)
    # 就像考试前：平时做练习题(Train)，最后做模拟考(Test)
    train_size = 40
    test_size = 10
    train_set, test_set = random_split(full_dataset, [train_size, test_size])

    # DataLoader: 负责把数据打包成 Batch
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

    # --- C. 模型与优化器初始化 ---
    model = UWB_CRNN_Optimized().to(device) # 把模型搬到显卡上
    
    # 损失函数 (Loss Function): 衡量“预测值”和“真值”差多少
    # BCELoss (Binary Cross Entropy) 专门用于 0-1 的概率输出
    criterion = nn.BCELoss() 
    
    # 优化器 (Optimizer): 负责调整参数 W
    # Adam 是目前最常用的自适应算法，lr 是学习率(步长)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- D. 训练循环 (Training Loop) ---
    epochs = 100 # 也就是把所有数据看 10 遍
    print("\n=== 开始训练 ===")
    
    for epoch in range(epochs):
        model.train() # 【重要】切换到训练模式 (启用 Dropout)
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 1. 搬运数据到显卡
            inputs, targets = inputs.to(device), targets.to(device)

            # 2. 梯度清零 (Zero Grad)
            # PyTorch 默认会累加梯度，每次反向传播前必须清零，否则梯度会乱
            optimizer.zero_grad()

            # 3. 前向传播 (Forward) -> 得到预测值
            outputs = model(inputs)

            # 4. 计算误差 (Compute Loss)
            loss = criterion(outputs, targets)

            # 5. 反向传播 (Backward) -> 计算梯度 (dLoss/dW)
            loss.backward()

            # 6. 参数更新 (Step) -> W_new = W_old - lr * gradient
            optimizer.step()

            total_loss += loss.item()

        # 打印这一轮的平均误差
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}")

    # --- E. 测试/验证循环 (Evaluation Loop) ---
    print("\n=== 开始测试 (验证集) ===")
    model.eval() # 【重要】切换到评估模式 (关闭 Dropout，冻结 BatchNormal)
    test_loss = 0
    
    # torch.no_grad(): 告诉 PyTorch 别算梯度了，省显存，反正测试时不更新参数
    with torch.no_grad(): 
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # 在这里，你可以添加代码把 outputs 保存下来画图
            # plt.imshow(outputs[0].cpu().numpy()) 

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print("模型训练完成，可以保存用于部署了。")

if __name__ == '__main__':
    main_pipeline()