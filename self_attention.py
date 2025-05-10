import torch
import torch.nn as nn
import math


# https://blog.csdn.net/weixin_44791964/article/details/121371986
# SENet是通道注意力机制的典型实现。
# 2017
# 年提出的SENet是最后一届ImageNet竞赛的冠军，其实现示意图如下所示，对于输入进来的特征层，我们关注其每一个通道的权重，对于SENet而言，其重点是获得输入进来的特征层，每一个通道的权值。利用SENet，我们可以让网络关注它最需要关注的通道。
#
# 其具体实现方式就是：
# 1、对输入进来的特征层进行全局平均池化。
# 2、然后进行两次全连接，第一次全连接神经元个数较少，第二次全连接神经元个数和输入特征层相同。
# 3、在完成两次全连接后，我们再取一次Sigmoid将值固定到0 - 1
# 之间，此时我们获得了输入特征层每一个通道的权值（0 - 1
# 之间）。
# 4、在获得这个权值后，我们将这个权值乘上原输入特征层即可。

class se_block(nn.Module):
	def __init__(self, channel, ratio=16):
		super(se_block, self).__init__()
		# 在高宽上进行全局平均池化 设置成1后  池化后高宽就是1了 b c h w -> b c 1 1
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // ratio, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // ratio, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		# b c h w -> b c 1 1 -> b c
		y = self.avg_pool(x).view(b, c)
		# b c -> b c 1 1
		y = self.fc(y).view(b, c, 1, 1)
		return x * y


# CBAM将通道注意力机制和空间注意力机制进行一个结合，相比于SENet只关注通道的注意力机制可以取得更好的效果。其实现示意图如下所示，CBAM会对输入进来的特征层，分别进行通道注意力机制的处理和空间注意力机制的处理。
#
# 下图是通道注意力机制和空间注意力机制的具体实现方式：
# 图像的上半部分为通道注意力机制，通道注意力机制的实现可以分为两个部分，我们会对输入进来的单个特征层，分别进行全局平均池化和全局最大池化。之后对平均池化和最大池化的结果，利用共享的全连接层进行处理，我们会对处理后的两个结果进行相加，然后取一个sigmoid，此时我们获得了输入特征层每一个通道的权值（0 - 1
# 之间）。在获得这个权值后，我们将这个权值乘上原输入特征层即可。
#
# 图像的下半部分为空间注意力机制，我们会对输入进来的特征层，在每一个特征点的通道上取最大值和平均值。之后将这两个结果进行一个堆叠，利用一次通道数为1的卷积调整通道数，然后取一个sigmoid，此时我们获得了输入特征层每一个特征点的权值（0 - 1
# 之间）。在获得这个权值后，我们将这个权值乘上原输入特征层即可。
class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=2):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)
		# 利用1x1卷积代替全连接
		self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# print(self.outc,self.in_p, self.in_p // self.ratio)
		# # 1 3 1 1
		# avg_out = self.avg_pool(x)
		# avg_out = self.fc1(avg_out)
		# avg_out = self.relu1(avg_out)
		# avg_out = self.fc2(avg_out)
		avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
		max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		out = avg_out + max_out
		return self.sigmoid(out)


class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7):
		super(SpatialAttention, self).__init__()

		assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
		padding = 3 if kernel_size == 7 else 1
		self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# 在通道上进行平均池化
		avg_out = torch.mean(x, dim=1, keepdim=True)
		# 在通道上进行最大池化
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		# dim = 1 代表按照在一维上拼接
		x = torch.cat([avg_out, max_out], dim=1)
		x = self.conv1(x)
		return self.sigmoid(x)


class cbam_block(nn.Module):
	def __init__(self, channel, ratio=2, kernel_size=7):
		super(cbam_block, self).__init__()
		self.channelattention = ChannelAttention(channel, ratio=ratio)
		self.spatialattention = SpatialAttention(kernel_size=kernel_size)

	def forward(self, x):
		x = x * self.channelattention(x)
		x = x * self.spatialattention(x)
		return x

# model = cbam_block(3)
# inputs = torch.ones([1,3,256,256])
# outputs = model(inputs)
# print(outputs)
# ECANet是也是通道注意力机制的一种实现形式。ECANet可以看作是SENet的改进版。
# ECANet的作者认为SENet对通道注意力机制的预测带来了副作用，捕获所有通道的依赖关系是低效并且是不必要的。
# 在ECANet的论文中，作者认为卷积具有良好的跨通道信息获取能力。
#
# ECA模块的思想是非常简单的，它去除了原来SE模块中的全连接层，直接在全局平均池化之后的特征上通过一个1D卷积进行学习。
#
# 既然使用到了1D卷积，那么1D卷积的卷积核大小的选择就变得非常重要了，了解过卷积原理的同学很快就可以明白，1
# D卷积的卷积核大小会影响注意力机制每个权重的计算要考虑的通道数量。用更专业的名词就是跨通道交互的覆盖率。
#
# 如下图所示，左图是常规的SE模块，右图是ECA模块。ECA模块用1D卷积替换两次全连接。

class eca_block(nn.Module):
	def __init__(self, channel, b=1, gamma=2):
		super(eca_block, self).__init__()
		kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
		kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
		y = self.sigmoid(y)
		return x * y.expand_as(x)



# import torch
#
# # 初始化三个 tensor
# A=torch.ones(2,3)    #2x3的张量（矩阵）
# # tensor([[ 1.,  1.,  1.],
# #         [ 1.,  1.,  1.]])
# B=2*torch.ones(4,3)  #4x3的张量（矩阵）
# # tensor([[ 2.,  2.,  2.],
# #         [ 2.,  2.,  2.],
# #         [ 2.,  2.,  2.],
# #         [ 2.,  2.,  2.]])
# D=2*torch.ones(2,4)	 # 2x4的张量（矩阵）
# # tensor([[ 2.,  2.,  2., 2.],
# #         [ 2.,  2.,  2., 2.],
#
# # 按维数0（行）拼接 A 和 B
# C=torch.cat((A,B),0)
# # tensor([[ 1.,  1.,  1.],
# #          [ 1.,  1.,  1.],
# #          [ 2.,  2.,  2.],
# #          [ 2.,  2.,  2.],
# #          [ 2.,  2.,  2.],
# #          [ 2.,  2.,  2.]])
# print(C.shape)
# # torch.Size([6, 3])
#
#
# # 按维数1（列）拼接 A 和 D
# C=torch.cat((A,D),1)
# # tensor([[ 1.,  1.,  1.,  2.,  2.,  2.,  2.],
# #         [ 1.,  1.,  1.,  2.,  2.,  2.,  2.]])
# print(C.shape)
# # torch.Size([2, 7])

# tensor([[1, 2], [3, 4]])
# “括号之间是嵌套关系，代表了不同的维度。从左往右数，两个括号代表的维度分别是 0 和 1 ，在第 0 维遍历得到向量，在第 1 维遍历得到标量”
