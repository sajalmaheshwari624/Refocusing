import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
from torchsummary import summary
from PIL import Image
from torchvision.models import vgg16

class EncoderDecoder(nn.Module) :
	def __init__(self, numChannelsIn, numChannelsOut) :
		super(EncoderDecoder, self).__init__()

		# Encoder block

		self.conv1 	= nn.Conv2d(in_channels = numChannelsIn, 	out_channels = 32, 	kernel_size = 7, stride = 2, padding = 3, bias = True)
		self.conv1b = nn.Conv2d(in_channels = 32, 	out_channels = 32, 	kernel_size = 7, stride = 1, padding = 3, bias = True)
		
		self.conv2 	= nn.Conv2d(in_channels = 32, 	out_channels = 64, 	kernel_size = 5, stride = 2, padding = 2, bias = True)
		self.conv2b = nn.Conv2d(in_channels = 64, 	out_channels = 64, 	kernel_size = 5, stride = 1, padding = 2, bias = True)
		
		self.conv3 	= nn.Conv2d(in_channels = 64, 	out_channels = 128, kernel_size = 3, stride = 2, padding = 1, bias = True)
		self.conv3b = nn.Conv2d(in_channels = 128, 	out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.conv4 	= nn.Conv2d(in_channels = 128, 	out_channels = 256, kernel_size = 3, stride = 2, padding = 1, bias = True)
		self.conv4b = nn.Conv2d(in_channels = 256, 	out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.conv5 	= nn.Conv2d(in_channels = 256, 	out_channels = 512, kernel_size = 3, stride = 2, padding = 1, bias = True)
		self.conv5b = nn.Conv2d(in_channels = 512, 	out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = True)

		# Decoder block
		
		self.upconv5 	= nn.ConvTranspose2d(in_channels = 512, 	out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = True)
		self.iconv5 	= nn.Conv2d(in_channels = 512, 	out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.upconv4 	= nn.ConvTranspose2d(in_channels = 256, 	out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = True)
		self.iconv4 	= nn.Conv2d(in_channels = 256, 	out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = True)
		self.disp4 		= nn.Conv2d(in_channels = 128, 	out_channels = numChannelsOut, 	kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.upconv3 	= nn.ConvTranspose2d(in_channels = 128, 	out_channels = 64, 	kernel_size = 4, stride = 2, padding = 1, bias = True)
		self.iconv3 	= nn.Conv2d(in_channels = 128 + numChannelsOut, 	out_channels = 64, 	kernel_size = 3, stride = 1, padding = 1, bias = True)
		self.disp3 		= nn.Conv2d(in_channels = 64, 	out_channels = numChannelsOut, 	kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.upconv2 	= nn.ConvTranspose2d(in_channels = 64, 	out_channels = 32, 	kernel_size = 4, stride = 2, padding = 1, bias = True)
		self.iconv2 	= nn.Conv2d(in_channels = 64 + numChannelsOut, 	out_channels = 32, 	kernel_size = 3, stride = 1, padding = 1, bias = True)
		self.disp2 		= nn.Conv2d(in_channels = 32, 	out_channels = numChannelsOut, 	kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.upconv1 	= nn.ConvTranspose2d(in_channels = 32, 	out_channels = 16, 	kernel_size = 4, stride = 2, padding = 1, bias = True)
		self.iconv1 	= nn.Conv2d(in_channels = 16 + numChannelsOut, 	out_channels = 16, 	kernel_size = 3, stride = 1, padding = 1, bias = True)
		self.disp1 		= nn.Conv2d(in_channels = 16, 	out_channels = numChannelsOut, 	kernel_size = 3, stride = 1, padding = 1, bias = True)

	def forward(self, inImage, segMask) :
		
		# Encoder Forward

		conv1Out 	= F.dropout(F.elu(self.conv1((torch.cat((inImage, segMask), dim = 1))), alpha = 1.0, inplace = False), p = 0.1)
		conv1bOut 	= F.dropout(F.elu(self.conv1b(conv1Out), 	alpha = 1.0, inplace = False), p = 0.1)

		conv2Out 	= F.dropout(F.elu(self.conv2(conv1bOut), 	alpha = 1.0, inplace = False), p = 0.1)
		conv2bOut 	= F.dropout(F.elu(self.conv2b(conv2Out), 	alpha = 1.0, inplace = False), p = 0.1)

		conv3Out 	= F.dropout(F.elu(self.conv3(conv2bOut), 	alpha = 1.0, inplace = False), p = 0.1)
		conv3bOut 	= F.dropout(F.elu(self.conv3b(conv3Out), 	alpha = 1.0, inplace = False), p = 0.1)
			
		conv4Out 	= F.dropout(F.elu(self.conv4(conv3bOut), 	alpha = 1.0, inplace = False), p = 0.1)
		conv4bOut 	= F.dropout(F.elu(self.conv4b(conv4Out), 	alpha = 1.0, inplace = False), p = 0.1)
			
		conv5Out 	= F.dropout(F.elu(self.conv5(conv4bOut), 	alpha = 1.0, inplace = False), p = 0.1)
		conv5bOut 	= F.dropout(F.elu(self.conv5b(conv5Out), 	alpha = 1.0, inplace = False), p = 0.1)
			

		# Decoder Forward

		upConv5Out 	= F.dropout(F.elu(self.upconv5(conv5bOut), alpha = 1.0, inplace = False), p = 0.1)
		iConv5Out 	= F.dropout(F.elu(self.iconv5(torch.cat((upConv5Out, conv4bOut), dim = 1)), alpha = 1.0, inplace = False), p = 0.1)

		upConv4Out 	= F.dropout(F.elu(self.upconv4(iConv5Out), alpha = 1.0, inplace = False), p = 0.1)
		iConv4Out 	= F.dropout(F.elu(self.iconv4(torch.cat((upConv4Out, conv3bOut), dim = 1)), alpha = 1.0, inplace = False), p = 0.1)
		disp4Out 	= torch.tanh(self.disp4(iConv4Out))

		upConv3Out 	= F.dropout(F.elu(self.upconv3(iConv4Out), alpha = 1.0, inplace = False), p = 0.1)
		iConv3Out 	= F.dropout(F.elu(self.iconv3(torch.cat((upConv3Out, conv2bOut, F.interpolate(disp4Out, size 	= None, scale_factor = 2.0, mode = 'bilinear')), dim = 1)), alpha = 1.0, inplace = False), p = 0.1)
		disp3Out 	= torch.tanh(self.disp3(iConv3Out))

		upConv2Out 	= F.dropout(F.elu(self.upconv2(iConv3Out), alpha = 1.0, inplace = False), p = 0.1)
		iConv2Out 	= F.dropout(F.elu(self.iconv2(torch.cat((upConv2Out, conv1bOut, F.interpolate(disp3Out, size 	= None, scale_factor = 2.0, mode = 'bilinear')), dim = 1)), alpha = 1.0, inplace = False), p = 0.1)
		disp2Out 	= torch.tanh(self.disp2(iConv2Out))

		upConv1Out 	= F.dropout(F.elu(self.upconv1(iConv2Out), alpha = 1.0, inplace = False), p = 0.1)
		iConv1Out 	= F.dropout(F.elu(self.iconv1(torch.cat((upConv1Out, F.interpolate(disp2Out, size = None, scale_factor = 2.0, mode = 'bilinear')), dim = 1)), alpha = 1.0, inplace = False), p = 0.1)
		disp1Out 	= torch.tanh(self.disp1(iConv1Out))

		return (disp1Out, disp2Out, disp3Out, disp4Out)

class EncoderDecoderv2(nn.Module) :
	def __init__(self, numChannelsIn, numChannelsOut) :
		super(EncoderDecoderv2, self).__init__()

		# Encoder block

		self.conv1 	= nn.Conv2d(in_channels = numChannelsIn, 	out_channels = 32, 	kernel_size = 7, stride = 2, padding = 3, bias = True)
		self.conv1b = nn.Conv2d(in_channels = 32, 	out_channels = 32, 	kernel_size = 7, stride = 1, padding = 3, bias = True)
		
		self.conv2 	= nn.Conv2d(in_channels = 32, 	out_channels = 64, 	kernel_size = 5, stride = 2, padding = 2, bias = True)
		self.conv2b = nn.Conv2d(in_channels = 64, 	out_channels = 64, 	kernel_size = 5, stride = 1, padding = 2, bias = True)
		
		self.conv3 	= nn.Conv2d(in_channels = 64, 	out_channels = 128, kernel_size = 3, stride = 2, padding = 1, bias = True)
		self.conv3b = nn.Conv2d(in_channels = 128, 	out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.conv4 	= nn.Conv2d(in_channels = 128, 	out_channels = 256, kernel_size = 3, stride = 2, padding = 1, bias = True)
		self.conv4b = nn.Conv2d(in_channels = 256, 	out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.conv5 	= nn.Conv2d(in_channels = 256, 	out_channels = 512, kernel_size = 3, stride = 2, padding = 1, bias = True)
		self.conv5b = nn.Conv2d(in_channels = 512, 	out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = True)

		# Decoder block
		
		self.upconv5 	= nn.ConvTranspose2d(in_channels = 512, 	out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = True)
		self.iconv5 	= nn.Conv2d(in_channels = 512, 	out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.upconv4 	= nn.ConvTranspose2d(in_channels = 256, 	out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = True)
		self.iconv4 	= nn.Conv2d(in_channels = 256, 	out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.upconv3 	= nn.ConvTranspose2d(in_channels = 128, 	out_channels = 64, 	kernel_size = 4, stride = 2, padding = 1, bias = True)
		self.iconv3 	= nn.Conv2d(in_channels = 128, 	out_channels = 64, 	kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.upconv2 	= nn.ConvTranspose2d(in_channels = 64, 	out_channels = 32, 	kernel_size = 4, stride = 2, padding = 1, bias = True)
		self.iconv2 	= nn.Conv2d(in_channels = 64, 	out_channels = 32, 	kernel_size = 3, stride = 1, padding = 1, bias = True)
		
		self.upconv1 	= nn.ConvTranspose2d(in_channels = 32, 	out_channels = 16, 	kernel_size = 4, stride = 2, padding = 1, bias = True)
		self.iconv1 	= nn.Conv2d(in_channels = 16, 	out_channels = 16, 	kernel_size = 3, stride = 1, padding = 1, bias = True)
		self.disp1 		= nn.Conv2d(in_channels = 16, 	out_channels = numChannelsOut, 	kernel_size = 3, stride = 1, padding = 1, bias = True)

	def forward(self, inImage) :
		
		# Encoder Forward

		conv1Out 	= F.dropout(F.elu(self.conv1(inImage), 	alpha = 1.0, inplace = False), p = 0.1)
		conv1bOut 	= F.dropout(F.elu(self.conv1b(conv1Out), 	alpha = 1.0, inplace = False), p = 0.1)

		conv2Out 	= F.dropout(F.elu(self.conv2(conv1bOut), 	alpha = 1.0, inplace = False), p = 0.1)
		conv2bOut 	= F.dropout(F.elu(self.conv2b(conv2Out), 	alpha = 1.0, inplace = False), p = 0.1)

		conv3Out 	= F.dropout(F.elu(self.conv3(conv2bOut), 	alpha = 1.0, inplace = False), p = 0.1)
		conv3bOut 	= F.dropout(F.elu(self.conv3b(conv3Out), 	alpha = 1.0, inplace = False), p = 0.1)
			
		conv4Out 	= F.dropout(F.elu(self.conv4(conv3bOut), 	alpha = 1.0, inplace = False), p = 0.1)
		conv4bOut 	= F.dropout(F.elu(self.conv4b(conv4Out), 	alpha = 1.0, inplace = False), p = 0.1)
			
		conv5Out 	= F.dropout(F.elu(self.conv5(conv4bOut), 	alpha = 1.0, inplace = False), p = 0.1)
		conv5bOut 	= F.dropout(F.elu(self.conv5b(conv5Out), 	alpha = 1.0, inplace = False), p = 0.1)
			

		# Decoder Forward

		upConv5Out 	= F.dropout(F.elu(self.upconv5(conv5bOut), alpha = 1.0, inplace = False), p = 0.1)
		iConv5Out 	= F.dropout(F.elu(self.iconv5(torch.cat((upConv5Out, conv4bOut), dim = 1)), alpha = 1.0, inplace = False), p = 0.1)

		upConv4Out 	= F.dropout(F.elu(self.upconv4(iConv5Out), alpha = 1.0, inplace = False), p = 0.1)
		iConv4Out 	= F.dropout(F.elu(self.iconv4(torch.cat((upConv4Out, conv3bOut), dim = 1)), alpha = 1.0, inplace = False), p = 0.1)

		upConv3Out 	= F.dropout(F.elu(self.upconv3(iConv4Out), alpha = 1.0, inplace = False), p = 0.1)
		iConv3Out 	= F.dropout(F.elu(self.iconv3(torch.cat((upConv3Out, conv2bOut), dim = 1)), alpha = 1.0, inplace = False), p = 0.1)

		upConv2Out 	= F.dropout(F.elu(self.upconv2(iConv3Out), alpha = 1.0, inplace = False), p = 0.1)
		iConv2Out 	= F.dropout(F.elu(self.iconv2(torch.cat((upConv2Out, conv1bOut), dim = 1)), alpha = 1.0, inplace = False), p = 0.1)

		upConv1Out 	= F.dropout(F.elu(self.upconv1(iConv2Out), alpha = 1.0, inplace = False), p = 0.1)
		iConv1Out 	= F.dropout(F.elu(self.iconv1(upConv1Out), alpha = 1.0, inplace = False), p = 0.1)
		disp1Out 	= torch.sigmoid(self.disp1(iConv1Out))

		return disp1Out

class DepthToInFocus(nn.Module) :
	def __init__(self, numOutChannels) :
		super(DepthToInFocus, self).__init__()

		self.conv1 = nn.Conv2d(in_channels = 9, out_channels = 60, kernel_size = 3, stride = 1, padding = 1, bias = True)
		self.convOut = nn.Conv2d(in_channels = 60, out_channels = numOutChannels, kernel_size = 3, stride = 1, padding = 1, bias = True)

	def forward(self, inImage, depthImage) :
		conv1Out = F.dropout(F.elu(self.conv1(torch.cat((inImage, depthImage), dim = 1)), alpha = 1.0, inplace = False), p = 0.1)
		inFocusOut = torch.tanh(self.convOut(conv1Out))
		return inFocusOut


if __name__ == "__main__" :
	device 				= torch.device("cuda:0")
	encoderDecoderModel1 	= EncoderDecoder(3,16).to(device)
	blurMapToInFocusModel 	= DepthToInFocus(3).to(device)
	encoderDecoderModel2 	= EncoderDecoderv2(3,3).to(device)
	summary(encoderDecoderModel, input_size=(3, 256, 256))
	summary(blurMapToInFocusModel, input_size=[(3, 256, 256), (1, 256, 256)])
	summary(encoderDecoderModel, input_size=(3, 256, 256))
#summary(blurMapToInFocusModel, input_size=[(3, 128, 128), (1, 128, 128)])
#summary(blurMapToInFocusModel, input_size=[(3, 64, 64), (1, 64, 64)])
#summary(blurMapToInFocusModel, input_size=[(3, 32, 32), (1, 32, 32)])


'''
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
	return (loss, SSIMLossTerm)

def getVGGOutput(inputImage, layers, device) :
	model = vgg16(pretrained = True).to(device)
	features = list(model.features)
	modFeatures = nn.ModuleList(features)

	x = inputImage 
	#reqList = {18, 27}
	results = []
	for ii,model in enumerate(modFeatures):
		#print (model)
		x = model(x)
		if ii in layers :
			mapDepth = int(x.size()[1])
			reqIndex = int(mapDepth / 2)
			reqMap = x[:,reqIndex, : , :]
			#print (torch.mean(reqMap))
			results.append(reqMap)
	return results


def perceptualLoss(outImage, inImage, layers, weights, device) :

	inFocusOutDispOut = inFocusOut.new(*inFocusOut.size())
	inFocusOutDispOut[:, 0, :, :] = inFocusOut[:, 0, :, :] * 0.5 + 0.5
	inFocusOutDispOut[:, 1, :, :] = inFocusOut[:, 1, :, :] * 0.5 + 0.5
	inFocusOutDispOut[:, 2, :, :] = inFocusOut[:, 2, :, :] * 0.5 + 0.5

	vggOutInImage = getVGGOutput(inImage, layers, device)
	vggOutOutImage = getVGGOutput(outImage, layers, device)

	vggDiffMeans = torch.zeros(2)
	loss = torch.zeros(1).to(device)

	for indices in range(len(layers)) :
		diffVGGOutputs = torch.mean((vggOutInImage[indices] - vggOutOutImage[indices]) ** 2)
		loss += weights[indices] * diffVGGOutputs
	#print (loss)
	return loss

# Loss testing code
im1Path = '/home/sajal/Desktop/BlurMap/data/InFocusLF/1.png'
im2Path = '/home/sajal/Desktop/BlurMap/data/NearFocusLF/1.png'
#ones = torch.ones(3, 256, 256)

im1 = Image.open(im1Path)
im2 = Image.open(im2Path)

trans = torchvision.transforms.ToPILImage()
trans1 = torchvision.transforms.ToTensor()

tensorIm1 = trans1(im1)
tensorIm2 = trans1(im2)

tensorIm1 = tensorIm1.view(1, 3, 256, 256)
tensorIm2 = tensorIm2.view(1, 3, 256, 256)

print (tensorIm1.size())

#tensorIm1 = torch.cat((tensorIm1, ones), dim = 0)
#tensorIm2 = torch.cat((tensorIm2, ones), dim = 0)

L1 = smoothnessLoss(tensorIm1, tensorIm2)
print (L1)

L2 = perceptualLoss(tensorIm1, tensorIm2, 0.85, 3)
print (L2)
'''
