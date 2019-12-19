import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dataset as DataUtils
import model
import losses
import helper
import time
#import matplotlib.pyplot as plt
from torchvision import transforms, datasets

def train(epochs) :
	imTransform = transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainLoader = helper.PreprocessInFocusOnlyData(imTransform, rootDirTrain, batchSize, True)
	validLoader = helper.PreprocessInFocusOnlyData(imTransform, rootDirValid, batchSize, True)
	slices =  ('outputSlice1','outputSlice4', 'outputSlice7', 'outputSlice10')
	numSlices = len(slices)


	rootDirToSave	= rootDirSave
	model1SaveDir 	= rootDirToSave + 'model1/'
	model2SaveDir 	= rootDirToSave + 'model2/'
	model3SaveDir 	= rootDirToSave + 'model3/'
	extension		= '.pth'

	loadRootDir 	= rootDirSave
	model1LoadPath 	= loadRootDir + 'model1/'
	model2LoadPath 	= loadRootDir + 'model2/'
	model3LoadPath 	= loadRootDir + 'model3/'

	model1 = model.EncoderDecoder(3, 6 * numSlices).to(device)
	model2 = model.DepthToInFocus(3).to(device)
	model3 = model.EncoderDecoderv2(3, 1).to(device)

	model1.load_state_dict(torch.load(model1LoadPath + str(loadEpochNum) + extension))
	model2.load_state_dict(torch.load(model2LoadPath + str(loadEpochNum) + extension))
	model3.load_state_dict(torch.load(model3LoadPath + str(loadEpochNum) + extension))
	
	contentLoss = losses.PerceptualLoss()
	contentLoss.initialize(nn.MSELoss())


	trainLossArray = np.zeros(epochs)
	validLossArray = np.zeros(epochs)

	lr = learningRate
	minValidEpoch = 0
	minValidLoss = 100.0

	for epoch in range(epochs) :
		start = time.clock()

		epochLoss = np.zeros(2)
		epochCount = epoch + 1

		print ("Epoch num : " + str(epochCount))
		if epoch >= 20 and epoch % 20 == 0 :
			lr = lr / 2
			print ('Learning rate changed to ' + str(lr))

		for step, sample in enumerate(trainLoader) :

			loss = torch.zeros(1).to(device)
			loss.requires_grad = False
			
			data 	= sample.to(device)
			
			blurOut = model1(data)

			for dispIndex in range(len(blurOut)) :

				scaleLoss = torch.zeros(1).to(device)
				scaleLoss.requires_grad = False

				dispOut 	= blurOut[dispIndex]

				inFocus	= F.interpolate(data, size = None, scale_factor = 1 / pow(2.0, dispIndex), mode = 'bilinear')

				blurMapSlices = helper.BlurMapsToSlices(dispOut, numSlices, 6)
				inFocusRegionSlices = []

				
				for slices in range(numSlices) :
				
					nearFocusOutSlice = model2(inFocus, blurMapSlices[slices])
					binaryBlurMapSlice = model3(nearFocusOutSlice)

					inFocusRegionSlice = torch.mul(nearFocusOutSlice, binaryBlurMapSlice)
					inFocusRegionSlices.append(inFocusRegionSlice)

				inFocusOutput = helper.CombineFocusRegionsToAIF(inFocusRegionSlices, numSlices, device)

				perceptLoss = contentLoss.get_loss(inFocus, inFocusOutput)
				simiLoss = losses.similarityLoss(inFocus, inFocusOutput, alpha, winSize)
				scaleLoss += perceptLossW * perceptLoss + simiLossW * simiLoss

				loss += scaleLoss
			
			epochLoss[0] += loss

			optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters()), lr=lr,\
			betas=(beta1, beta2), eps=1e-08, weight_decay=0, amsgrad=False)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if step % 500 == 0 :
					print ("Step num :" + str(step) + " Train Loss :", str(loss.item()))

		with torch.no_grad() :
			for validStep, validSample in enumerate(validLoader) :

				validLoss = torch.zeros(1).to(device)
				validLoss.requires_grad = False
			
				validData 	= sample.to(device)
			
				validBlurOut = model1(validData)

				for validDispIndex in range(len(validBlurOut)) :

					validScaleLoss = torch.zeros(1).to(device)
					validScaleLoss.requires_grad = False

					validDispOut 	= validBlurOut[validDispIndex]

					validInFocus	= F.interpolate(validData, size = None, scale_factor = 1 / pow(2.0, validDispIndex), mode = 'bilinear')

					validBlurMapSlices = helper.BlurMapsToSlices(validDispOut, numSlices, 6)
					validInFocusRegionSlices = []

				
					for validSlices in range(numSlices) :

						validNearFocusOutSlice = model2(validInFocus, validBlurMapSlices[validSlices])
						validBinaryBlurMapSlice = model3(validNearFocusOutSlice)
						validInFocusRegionSlice = torch.mul(validNearFocusOutSlice, validBinaryBlurMapSlice)
						validInFocusRegionSlices.append(validInFocusRegionSlice)

					validInFocusOutput = helper.CombineFocusRegionsToAIF(validInFocusRegionSlices, numSlices, device)

					validPerceptLoss = contentLoss.get_loss(validInFocus, validInFocusOutput)
					validSimiLoss = losses.similarityLoss(validInFocus, validInFocusOutput, alpha, winSize)
					validScaleLoss += perceptLossW * validPerceptLoss + simiLossW * validSimiLoss

					validLoss += validScaleLoss
			
				epochLoss[1] += validLoss

		epochLoss[0] /= step
		epochLoss[1] /= validStep
		epochFreq = epochSaveFreq
		print ("Time taken for epoch " + str(epoch) + " is " + str(time.clock() - start))
		
		if validLoss < minValidLoss :
			minValidLoss = validLoss
			minValidEpoch = epoch
			torch.save(model1.state_dict(), model1SaveDir + str(epoch) + 'generic_' + extension)
			torch.save(model2.state_dict(), model2SaveDir + str(epoch) + 'generic_' + extension)
			torch.save(model3.state_dict(), model3SaveDir + str(epoch) + 'generic_' + extension)
			print ("Saving latest model at epoch: " + str(epoch))

		if epochCount % epochFreq == 0 :
			torch.save(model1.state_dict(), model1SaveDir + str(epoch) + 'generic_' + extension)
			torch.save(model2.state_dict(), model2SaveDir + str(epoch) + 'generic_' + extension)
			torch.save(model3.state_dict(), model3SaveDir + str(epoch) + 'generic_' + extension)
					
		print ("Training loss for epoch " + str(epochCount) + " : " + str(epochLoss[0]))
		print ("Validation loss for epoch " + str(epochCount) + " : " + str(epochLoss[1]))

		trainLossArray[epoch] = epochLoss[0]
		validLossArray[epoch] = epochLoss[1]
	
	print (minValidEpoch)
	return (trainLossArray, validLossArray)

epochs 			= 50
loadEpochNum 	= 41
batchSize 		= 2
epochSaveFreq	= 5
device 			= torch.device("cuda:0")
alpha			= 0.85
winSize			= 3
simiLossW	= 1.0
smoothLossW		= 1.0
perceptLossW = 0.1
learningRate	= 0.0001
jointTraining = True
beta1			= 0.9
beta2			= 0.999

rootDirTrain    = '/home/sajal.maheshwari/BlurMapExp/PortraitData/trainData/'
rootDirValid    = '/home/sajal.maheshwari/BlurMapExp/PortraitData/validData/'
rootDirSave     = '/home/sajal.maheshwari/BlurMapExp/BlurMapv2/checkpoints/'
	
(trainLossArray, validLossArray) = train(epochs)
lossGraph = plt.plot(trainLossArray)
lossGraph = plt.plot(validLossArray)
