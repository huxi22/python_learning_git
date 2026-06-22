import numpy as np
import torch

x = np.array([[1, 2, 3, 4, 5.5],
              [1, 2, 3, 4, 5.5]], dtype=float)
print(x.shape)

y = torch.tensor(x, dtype=torch.float32, requires_grad=True)
print(y.shape)

y_flat = y.reshape(-1, 1)
print(f"{y_flat.shape} on {y_flat.device}")


# 正常深度学习训练场景下，我们只需要叶子节点（模型可训练参数）的梯度，不需要给中间节点加 retain_grad()，
# 它会额外占用显存。只有调试、需要查看中间层梯度时才会用到这个方法。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y_flat = y_flat.to(device)                      
print(f"{y_flat.shape} on {y_flat.device}")


z1 = y_flat[0, 0]**2 + 0.5*y_flat[1, 0]**2 + 0.1*y_flat[2, 0]**2
z1.backward(retain_graph=True)
print(y.grad)

z2 = y_flat[0, 0]**2 + y_flat[1, 0]**2 + y_flat[2, 0]**2
z2.backward()
print(y.grad)


# 如果Tensor在GPU上，需要先转回CPU；如果这个Tensor参与了梯度计算，还要先detach；
x_back = y.detach().cpu().numpy()
print(x_back.shape)


# x = torch.tensor([1.0, 2.0, 3.4], requires_grad=True)
# y1 = x[0]**2 + 0.5*x[1]**2 + 0.1*x[2]**2
# y2 = x[0]**2 + x[1]**2 + x[2]**2

# y1.backward(retain_graph=True)
# print(x.grad)

# # x.grad.zero_()
# y2.backward()
# print(x.grad)






