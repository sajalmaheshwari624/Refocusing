import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dataset as DataUtils
import model
import helper
import pytorch_ssim
#import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.utils import save_image

def test() :

	imTransform = transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	testLoader = helper.PreprocessInFocusOnlyData(imTransform, rootDirTest, 'inputs', 'masks', batchSize, False)
	sliceNames =  ('outputSlice1','outputSlice10')
	numSlices = len(sliceNames)

	resultInputDir = rootDirResults + 'InputAIF/'
	resultDirInFocus = rootDirResults + 'OutputAIF/'
	resultDirSlices = rootDirResults + 'SliceOutputs/'
	resultDirSegMasks = rootDirResults + 'GroundTruthMasks/'

	extension	= '.pth'

	loadRootDir 	= rootDirLoad
	epoch 			= loadEpochNum
	model1LoadPath 	= loadRootDir + 'model1/'
	model2LoadPath 	= loadRootDir + 'model2/'
	model3LoadPath 	= loadRootDir + 'model3/'

	print (model1LoadPath, model2LoadPath, model3LoadPath)

	model1 = model.EncoderDecoder(4, 6 * numSlices).to(device)
	model1.load_state_dict(torch.load(model1LoadPath + str(epoch) + extension))
	model1.eval()


	model2 = model.DepthToInFocus(3).to(device)
	model2.load_state_dict(torch.load(model2LoadPath + str(epoch) + extension))
	model2.eval()

	model3 = model.EncoderDecoderv2(3, 1).to(device)
	model3.load_state_dict(torch.load(model3LoadPath + str(epoch) + extension))
	model3.eval()

	count = 0
	
	for step, sample in enumerate(testLoader) :
		print (step)
		inData  = sample['Input'].to(device)
		print (inData.shape)
		segMask = sample['Mask'].to(device)
		blurOut = model1(inData, segMask)
		count += 1

		for dispIndex in range(len(blurOut)) :

			dispOut = blurOut[dispIndex]

			inFocus	= F.interpolate(inData, size = None, scale_factor = 1 / pow(2.0, dispIndex), mode = 'bilinear')

			blurMapSlices = helper.BlurMapsToSlices(dispOut, numSlices, 6)
			inFocusRegionSlices = []
			nearFocusOutSlices = []
			binaryBlurMapSlices = []

				
			for slices in range(numSlices) :

				nearFocusOutSlice = model2(inFocus, blurMapSlices[slices])
				binaryBlurMapSlice = model3(nearFocusOutSlice)
				binaryBlurMapSlice = torch.cat((binaryBlurMapSlice, binaryBlurMapSlice, binaryBlurMapSlice), dim = 1)
				inFocusRegionSlice = torch.mul(nearFocusOutSlice, binaryBlurMapSlice)
				inFocusRegionSlices.append(inFocusRegionSlice)
				nearFocusOutSlices.append(nearFocusOutSlice)
				binaryBlurMapSlices.append(binaryBlurMapSlice)

			inFocusOutput = helper.CombineFocusRegionsToAIF(inFocusRegionSlices, numSlices, device)

			if dispIndex == 0 :

				inFocusOutDispOut = helper.TensorToDispImage(inFocusOutput, 3)
				inDataDispOut = helper.TensorToDispImage(inFocus, 3)
				inSegMask = F.interpolate(segMask, size = None, scale_factor = 1 / pow(2.0, dispIndex), mode = 'bilinear')

				save_image(inFocusOutDispOut, resultDirInFocus + str(step) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
				save_image(inDataDispOut, resultInputDir + str(step) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
				save_image(inSegMask, resultDirSegMasks + str(step) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

				for indices in range(numSlices) :
					
					dirToWrite = resultDirSlices + sliceNames[indices] + '/'
					sliceDir = dirToWrite + 'FocalSlices/'
					blurMapDir = dirToWrite + 'BlurMaps/'
					binaryMapDir = dirToWrite + 'BinaryMaps/'
					inFocusRegionDir = dirToWrite + 'InFocusRegion/'

					focalSliceDispOut = helper.TensorToDispImage(nearFocusOutSlices[indices], 3)
					binaryMapDispOut = binaryBlurMapSlices[indices]
					inFocusRegionDispOut = helper.TensorToDispImage(inFocusRegionSlices[indices], 3)

					displayOut = blurMapSlices[indices].mean(1)
					displayOutDispOut = displayOut.new(*displayOut.size())
					displayOutDispOut[0, :, :] = displayOut[0, :, :] * 0.5 + 0.5

					save_image(focalSliceDispOut, sliceDir + str(step) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
					save_image(displayOutDispOut, blurMapDir + str(step) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
					save_image(binaryMapDispOut, binaryMapDir + str(step) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
					save_image(inFocusRegionDispOut, inFocusRegionDir+ str(step) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)


batchSize 		= 1
device 			= torch.device("cuda:0")
loadEpochNum	= 152
goldenTest	= False

'''
#/home/sajal.maheshwari/BlurMapExp/
if goldenTest == False :
	#rootDirTest		= '/home/sajal.maheshwari/BlurMapExp/SelfieDataAIF/testData/'
	#rootDirLoad		= '/home/sajal.maheshwari/BlurMapExp/Baseline2SlicesInterleaved/checkpoints/'
	#rootDirResults	= '/home/sajal.maheshwari/BlurMapExp/Baseline2SlicesInterleaved/selfieResults/'
	rootDirTest             = '/home/sajal/Desktop/SelfieDataAIF/eccvData/'
	rootDirLoad             = '/home/sajal/Desktop/BufferResutlsFinal/Baseline2SlicesInterleaved/checkpoints/'
	rootDirResults  = '/home/sajal/Desktop/BufferResutlsFinal/Baseline2SlicesInterleaved/eccvResult/'


else :
	rootDirTest             = '/home/sajal.maheshwari/BlurMapExp/goldenRealData/'
	rootDirLoad             = '/home/sajal.maheshwari/BlurMapExp/BlurMap/checkpoints/'
	rootDirResults  = '/home/sajal.maheshwari/BlurMapExp/BlurMap/goldenResults/'
'''

with torch.no_grad() :
	test()
