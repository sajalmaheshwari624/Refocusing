import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
from torchsummary import summary
from PIL import Image
from torchvision.models import vgg19
from torchvision import transforms

'''
class LossNetwork(nn.Module) :
	def __init__(self, vgg_model):
		super(LossNetwork, self).__init__()
		self.vgg_layers = vgg_model.features
		self.layers = {23}

	def forward(self, x) :
		results = []
		xUnNorm = x.new(*x.size())
		xUnNorm[:, 0, :, :] = (((x[:, 0, :, :] * 0.5 + 0.5) - 0.485) / 0.229)
		xUnNorm[:, 1, :, :] = (((x[:, 1, :, :] * 0.5 + 0.5) - 0.456) / 0.224)
		xUnNorm[:, 2, :, :] = (((x[:, 2, :, :] * 0.5 + 0.5) - 0.405) / 0.225)

		for index, module in self.vgg_layers._modules.items() :
			intIndex = int(index)
			x = module(x)
			count = 0
			if intIndex in self.layers :
				mapDepth = x.size()[1]
				reqIndex = int(mapDepth / 2)
				#print (mapDepth, reqIndex)
				reqMap = x[:,reqIndex, : , :]
				#print (torch.mean(reqMap))
				results.append(reqMap)
		return results
'''
class PerceptualLoss():
	
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def initialize(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss

#device = torch.device("cuda:0")
#vgg_model = vgg16(pretrained = True).to(device)
#loss_network = LossNetwork(vgg_model)
#loss_network.eval()
#t = torch.ones([1, 3, 256, 256]).to(device)
#o = loss_network(t)
#print (o)

def smoothnessLoss(depthImage, inImage) :
	delXdepth 		= torch.abs(depthImage[:, :, :, 1:] - depthImage[:, :, :, :-1])
	delXinImage 	= torch.abs(inImage[:, :, :, 1:] 	- inImage[:, :, :, :-1])
	delYdepth 		= torch.abs(depthImage[:, :, 1:, :] - depthImage[:, :, :-1, :])
	delYinImage 	= torch.abs(inImage[:, :, 1:, :] 	- inImage[:, :, :-1, :])

	#print (delXdepth.size())
	#print (delXinImage.size())
	#print (delYdepth.size())
	#print (delYinImage.size())

	expDelXInImage = torch.exp(-delXinImage)
	expDelYInImage = torch.exp(-delYinImage)

	#print (expDelXInImage.size())
	#print (expDelYInImage.size())

	expDelXInImageMag = torch.norm(expDelXInImage, p = 'fro', dim = 1, keepdim = True)
	expDelYInImageMag = torch.norm(expDelYInImage, p = 'fro', dim = 1, keepdim = True)

	#print (expDelXInImageMag.size())
	#print (expDelYInImageMag.size())

	loss = torch.mean(torch.mul(expDelXInImageMag, delXdepth)) + torch.mean(torch.mul(expDelYInImageMag, delYdepth))
	#loss = torch.mean(delXdepth) + torch.mean(delYdepth)
	#print (loss)
	return loss

def similarityLoss(outImage, inImage, alpha, windowSize) :
	L1LossTerm 		= F.l1_loss(outImage, inImage, reduction = 'mean')
	SSIMLossTerm 	= pytorch_ssim.ssim(outImage, inImage, window_size = windowSize, size_average = True)

	loss = alpha * ((1 - SSIMLossTerm) / 2) + (1 - alpha) * L1LossTerm
	#print (loss)
	return loss

def gradientLossBasic(outImage, inImage) :
	xfilter = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	yfilter = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

	xfilter = xfilter.view((1, 1, 3, 3))
	yfilter = yfilter.view((1, 1, 3, 3))

	xfilter = torch.cat((xfilter, xfilter, xfilter), dim = 1).cuda()
	yfilter = torch.cat((yfilter, yfilter, yfilter), dim = 1).cuda()

	xGradIn = F.conv2d(inImage, xfilter)
	yGradIn = F.conv2d(inImage, yfilter)

	xGradOut = F.conv2d(outImage, xfilter)
	yGradOut = F.conv2d(outImage, yfilter)

	gradLossTermx = F.mse_loss(xGradIn, xGradOut, reduction = 'mean')
	gradLossTermy = F.mse_loss(yGradIn, yGradOut, reduction = 'mean')

	loss = (gradLossTermx + gradLossTermy) / 2
	return loss

def binarizationLoss(inImage) :
	diffImage = inImage - 0.5
	squaredDiff = torch.mul(diffImage, diffImage)
	loss = torch.mean(squaredDiff)
	return loss	

'''

def gradientLossCanny(outImage, inImage) :
	net = net_canny.Net(threshold=3.0, use_cuda = True).cuda()
	net.eval()
	_, _, _, _,inCanny,_ = net(inImage)
	_, _, _, _,outCanny,_ = net(outImage)
	loss = F.mse_loss(inCanny, outCanny, reduction = 'mean')
	return loss
'''
	
'''
def perceptualLoss(outImage, inImage) :
	loss = torch.zeros(1)
	for indices in range(len(weights)) :
		diffVGGOutputs = torch.mean((outImage[indices] - inImage[indices]) ** 2)
		loss += weights[indices] * diffVGGOutputs
	#print (len(outImage))
	#print (len(inImage))
	loss = torch.mean((outImage[0] - inImage[0]) ** 2)
	return loss
'''
