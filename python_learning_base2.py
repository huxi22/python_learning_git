import numpy as np
import torch

# a = np.array([1, 2, 3])
# b = np.array([
#     [1, 2, 3], 
#     [4, 5, 6]
# ])

# print(a.shape)
# print(b.shape)


# x = np.zeros((3, 4))
# x = np.ones((3, 4))
# x = np.random.randn(64, 1, 28, 28)
# y = x.reshape(64, 1, 784)
# print(x.shape)
# print(y.shape)


x = np.array([
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7]
])
y = np.array([
    [1, 2, 3],
    [2, 3, 4]
])
x = x.transpose()
z = x@y
print(z)
print(z.std(axis=0))






# a_ori = torch.randn(10, 10, 64)
# a_flat1 = a_ori.view(a_ori.size(0), -1)
# a_flat2 = a_ori.reshape(20, -1)
# print(a_flat1.shape)
# print(a_flat2.shape)



# b_ori = torch.randn(10, 32, 100)
# b_trans = b_ori.permute(0, 2, 1)
# b_flat = b_trans.contiguous().view(-1, b_trans.size(2))
# print(b_ori.shape)
# print(f"permute后的size为{b_trans.shape}，首地址为{b_trans.data_ptr()}")
# print(f"reshape后的size为{b_flat.shape}，首地址为{b_flat.data_ptr()}")



# c_ori  = torch.randn(10, 100, 32)
# c_flat = c_ori.reshape(-1, c_ori.size(2))
# c_flat = c_flat.unsqueeze(1)
# c_flat = c_flat.unsqueeze(3)
# print(c_flat.shape) 

# c_flat = c_flat.squeeze(2)
# print(c_flat.shape) 


