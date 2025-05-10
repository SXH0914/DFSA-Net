import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from layers import *




class SCM(nn.Module):
	def __init__(self, out_plane):
		super(SCM, self).__init__()
		self.main = nn.Sequential(
			BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
			BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
			BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
			BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
			nn.InstanceNorm2d(out_plane, affine=True)
		)

	def forward(self, x):
		x = self.main(x)
		return x


class FAM(nn.Module):
	def __init__(self, channel):
		super(FAM, self).__init__()
		self.merge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, relu=False)

	def forward(self, x1, x2):
		return self.merge(torch.cat([x1, x2], dim=1))


class ConvIR(nn.Module):
	def __init__(self, num_res):
		super(ConvIR, self).__init__()


		num_res = 4


		base_channel = 3


		self.feat_extract = nn.ModuleList([
			BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
			BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
			BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
			BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
			BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
			BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
		])


		self.Convs = nn.ModuleList([
			BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
			BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
		])

		self.ConvsOut = nn.ModuleList(
			[
				BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
				BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
			]
		)

		self.FAM1 = FAM(base_channel * 4)
		self.SCM1 = SCM(base_channel * 4)
		self.FAM2 = FAM(base_channel * 2)
		self.SCM2 = SCM(base_channel * 2)
		self.cbam_6 = cbam_block(6)
		self.cbam_12 = cbam_block(12)
		self.cbam_3 = cbam_block(3)

	def forward(self, x):
		# x_2 1 3 128 128
		x_2 = F.interpolate(x, scale_factor=0.5)
		# torchvision.utils.save_image(x_2, "128.jpg")
		# x_4 1 3 64 64
		x_4 = F.interpolate(x_2, scale_factor=0.5)

		# 通道扩张
		# z2 1 6 128 128
		z2 = self.SCM2(x_2)
		z2 = self.cbam_6(z2)

		# z4 1 12 64 64
		z4 = self.SCM1(x_4)
		z4 = self.cbam_12(z4)

		outputs = list()
		# 256
		# 1 3 256 256
		x_256 = self.feat_extract[0](x)
		# torchvision.utils.save_image(x_256, "n256.jpg")
		x_256 = self.cbam_3(x_256)

		# 128
		# 1 6 128 128
		z = self.feat_extract[1](x)
		# # 注意力
		z = self.cbam_6(z)
		# 1 6 128 128
		z = self.FAM2(z, z2)
		z = self.cbam_6(z)

		# 1 3 128 128
		z_128 = self.ConvsOut[1](z)


		# 64
		# 1 12 64 64
		z = self.feat_extract[2](z)
		# # 注意力
		z = self.cbam_12(z)
		# 1 12 64 64
		z = self.FAM1(z, z4)
		z = self.cbam_12(z)
		# 1 3 64 64
		z_64 = self.ConvsOut[0](z)

		# 开始恢复
		# 128
		# 1 6 128 128
		z = self.feat_extract[3](z)
		# # 注意力
		z = self.cbam_6(z)
		# 1 6 128 128
		z = self.FAM2(z, z2)
		z = self.cbam_6(z)
		# 恢复64的
		# 1 3 64 64
		outputs.append(z_64 + x_4)


		# 256
		# 1 3 256 256
		z = self.feat_extract[4](z)
		# # 注意力
		z = self.cbam_3(z)
		# 恢复 128 的
		outputs.append(z_128 + x_2)
		# 1 3 256 256
		z = self.feat_extract[5](z)
		z = self.cbam_3(z)
		outputs.append(z + x_256)

		return outputs


def build_net(version, data):
	return ConvIR(version, data)
