from Unet import *
from Convf import *
from DCP import *

class ALLnet(nn.Module):
	def __init__(self):
		super(ALLnet, self).__init__()
		# 频率盒子
		self.frequence_cube = GLGenerator(r=21, eps=0.01)
		# dcp
		self.pre_net = DCPGenerator(win_size=5, r=15, eps=0.001)
		self.low_model = ConvIR(4)
		self.high_model = UNet()

	def forward(self, x):


		# dcp
		(I_dcp, b) = self.pre_net(x)


		# 频率盒子
		low_frequence, high_frequence = self.frequence_cube(I_dcp)

		high_frequence = torch.cat((high_frequence, x), dim=1)


		low = self.low_model(low_frequence)[2]
		high =  self.high_model(high_frequence)

		out = low + high

		return out


if __name__ == "__main__":
	net = ALLnet().cuda()
	x = torch.randn(1,3,256,256).cuda()
	out = net(x)
	print(out.shape)