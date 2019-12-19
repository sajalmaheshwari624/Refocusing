import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dataset as DataUtils
import model
import losses
#import matplotlib.pyplot as plt
from torchvision import transforms, datasets

def PreprocessData(imageTransforms, slices, rootDir, inputDir, segmDir, batchSize, shuffleFlag) :
	#self, transforms, root_dir, output_dirs, input_dir, segm_dir
	dataLoad = DataUtils.Data(imageTransforms, root_dir = rootDir, output_dirs = slices, input_dir = inputDir, segm_dir = segmDir)
	dataLoader = torch.utils.data.DataLoader(dataLoad, batch_size = batchSize, shuffle = shuffleFlag)
	return dataLoader
#self, transforms, root_dir, input_dir, mask_dir
def PreprocessInFocusOnlyData(imageTransforms, rootDir, inputDir, maskDir, batchSize, shuffleFlag) :
	dataLoad = DataUtils.InFocusData(imageTransforms, root_dir = rootDir, input_dir = inputDir, mask_dir = maskDir)
	dataLoader = torch.utils.data.DataLoader(dataLoad, batch_size = batchSize, shuffle = shuffleFlag)
	return dataLoader

def BlurMapsToSlices(blurMaps, numSlices, blurChannels) :
	slicedMaps = []
	for index in range(0, numSlices * blurChannels, blurChannels) :
		slicedMaps.append(blurMaps[:, index: index + blurChannels, :, :])
	return slicedMaps

def FocalStackToSlices(focalStack, numSlices) :
	focalSlices = []
	for index in range(0, numSlices * 3, 3) :
		focalSlices.append(focalStack[:, index: index + 3, :, :])
	return focalSlices

def CombineFocusRegionsToAIF(inFocusSlices, numSlices, device) :
	channels = 3
	height = inFocusSlices[0].shape[2]
	width = inFocusSlices[0].shape[3]
	batchSize = inFocusSlices[0].shape[0]
	inFocusImage = torch.zeros([batchSize, channels, height, width]).to(device)
	for index  in range(0, numSlices) :
		inFocusImage += inFocusSlices[index]
	#inFocusImage = inFocusImage / numSlices
	return inFocusImage

def TensorToDispImage(inTensor, channels) :
	if channels == 3 :
		dispOut = inTensor.new(*inTensor.size())
		dispOut[:, 0, :, :] = inTensor[:, 0, :, :] * 0.5 + 0.5
		dispOut[:, 1, :, :] = inTensor[:, 1, :, :] * 0.5 + 0.5
		dispOut[:, 2, :, :] = inTensor[:, 2, :, :] * 0.5 + 0.5

	return dispOut
