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
	sliceNames =  ('outputSlice1','outputSlice10')
	numSlices = len(sliceNames)

	trainLoaderLF = helper.PreprocessData(imTransform, sliceNames, rootDirTrainLF, 'input', 'segments', batchSize, True)
	validLoaderLF = helper.PreprocessData(imTransform, sliceNames, rootDirValidLF, 'input', 'segments', batchSize, True)

	trainLoaderReal = helper.PreprocessInFocusOnlyData(imTransform, rootDirTrainReal, 'inputs', 'masks', batchSize, False)
	validLoaderReal = helper.PreprocessInFocusOnlyData(imTransform, rootDirValidReal, 'inputs', 'masks', batchSize, False)



	rootDirToSave	= rootDirSave
	model1SaveDir 	= rootDirToSave + 'model1/'
	model2SaveDir 	= rootDirToSave + 'model2/'
	model3SaveDir 	= rootDirToSave + 'model3/'
	extension		= '.pth'

	model1 = model.EncoderDecoder(4, 6 * numSlices).to(device)
	model2 = model.DepthToInFocus(3).to(device)
	model3 = model.EncoderDecoderv2(3, 1).to(device)
	
	contentLoss = losses.PerceptualLoss()
	contentLoss.initialize(nn.MSELoss())


	trainLossArray = np.zeros(epochs)
	validLossArray = np.zeros(epochs)

	lr = learningRate
	minValidEpoch = 0
	minValidLoss = 100.0

	trainLoaderRealList = list(trainLoaderReal)
	validLoaderRealList = list(validLoaderReal)
	numRealSteps = len(trainLoaderRealList)

	for epoch in range(epochs) :
		start = time.clock()

		realStep = 0

		epochLoss = np.zeros(2)
		epochCount = epoch + 1

		print ("Epoch num : " + str(epochCount))
		if epoch >= 20 and epoch % 20 == 0 :
			lr = lr / 2
			print ('Learning rate changed to ' + str(lr))
		if epoch < 40 :
			binarizationW = 0.0

		for step, sample in enumerate(trainLoaderLF) :

			if step % 3 == 1 and realStep < numRealSteps:
				realStepLoss = torch.zeros(1).to(device)
				realInData = trainLoaderRealList[realStep]['Input'].to(device)
				realSegMask = trainLoaderRealList[realStep]['Mask'].to(device)
				realBlurOut = model1(realInData, realSegMask)
				for realDispIndex in range(len(realBlurOut)) :
					realScaleLoss = torch.zeros(1).to(device)
					realBinarizationLoss = torch.zeros(1).to(device)
					realDispOut = realBlurOut[realDispIndex]
					realInFocus = F.interpolate(realInData, size = None, scale_factor = 1/pow(2.0, realDispIndex), mode = 'bilinear')
					realBlurMapSlices = helper.BlurMapsToSlices(realDispOut, numSlices, 6)
					realInFocusRegionSlices = []
					for realSlices in range(numSlices) :
						realNearFocusOutSlice = model2(realInFocus, realBlurMapSlices[realSlices])
						realBinaryBlurMapSlice = model3(realNearFocusOutSlice)
						realBinaryBlurMapSlice = torch.cat((realBinaryBlurMapSlice, realBinaryBlurMapSlice, realBinaryBlurMapSlice), dim = 1)
						binaryLoss = losses.binarizationLoss(realBinaryBlurMapSlice) * binarizationLossW
						realBinarizationLoss += binaryLoss
						realInFocusRegionSlice = torch.mul(realNearFocusOutSlice, realBinaryBlurMapSlice)
						realInFocusRegionSlices.append(realInFocusRegionSlice)
					realInFocusOutput = helper.CombineFocusRegionsToAIF(realInFocusRegionSlices, numSlices, device)
					realPerceptLoss = contentLoss.get_loss(realInFocus, realInFocusOutput)
					realSimiLoss = losses.similarityLoss(realInFocus, realInFocusOutput, alpha, winSize)
					realScaleLoss += perceptLossW * realPerceptLoss + simiLossW * realSimiLoss + realBinarizationLoss
					realStepLoss += realScaleLoss
					loss += realStepLoss
				optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters()), lr=lr/5,\
				betas=(beta1, beta2), eps=1e-08, weight_decay=0, amsgrad=False)
				optimizer.zero_grad()
				realStepLoss.backward(retain_graph = True)
				optimizer.step()
				realStep += 1

			stepLossStage1 = torch.zeros(1).to(device)
			stepLossStage1.requires_grad = False

			stepLossStage2 = torch.zeros(1).to(device)
			stepLossStage2.requires_grad = False

			loss = torch.zeros(1).to(device)
			loss.requires_grad = False
			
			inData 	= sample['Input'].to(device)
			outData = sample['Output'].to(device)
			inSegMask = sample['Mask'].to(device)

			blurOut = model1(inData, inSegMask)

			for dispIndex in range(len(blurOut)) :

				scaleLossStage1 = torch.zeros(1).to(device)
				scaleLossStage1.requires_grad = False

				scaleLossStage2 = torch.zeros(1).to(device)
				scaleLossStage2.requires_grad = False

				dispOut 	= blurOut[dispIndex]

				inFocus	= F.interpolate(inData, size = None, scale_factor = 1 / pow(2.0, dispIndex), mode = 'bilinear')
				nearFocusStackGT 	= F.interpolate(outData, size = None, scale_factor = 1 / pow(2.0, dispIndex) , mode = 'bilinear')

				nearFocusGTSlices = helper.FocalStackToSlices(nearFocusStackGT, numSlices)
				blurMapSlices = helper.BlurMapsToSlices(dispOut, numSlices, 6)
				inFocusRegionSlices = []

				
				for slices in range(numSlices) :

					sliceLoss = torch.zeros(1).to(device)
					sliceLoss.requires_grad = False
					#print (time.clock() - start)
				
					nearFocusOutSlice = model2(inFocus, blurMapSlices[slices])
					#print (time.clock() - start)

					perceptLoss = contentLoss.get_loss(nearFocusGTSlices[slices], nearFocusOutSlice)
					simiLoss = losses.similarityLoss(nearFocusGTSlices[slices], nearFocusOutSlice, alpha, winSize)

					sliceLoss = perceptLossW * perceptLoss + simiLoss * simiLoss

					binaryBlurMapSlice = model3(nearFocusOutSlice)
					binaryBlurMapSlice = torch.cat((binaryBlurMapSlice, binaryBlurMapSlice, binaryBlurMapSlice), dim = 1)
					binaryLoss = losses.binarizationLoss(binaryBlurMapSlice) * binarizationLossW
					sliceLoss += binaryLoss
					inFocusRegionSlice = torch.mul(nearFocusOutSlice, binaryBlurMapSlice)
					inFocusRegionSlices.append(inFocusRegionSlice)

					scaleLossStage1 += sliceLoss

				inFocusOutput = helper.CombineFocusRegionsToAIF(inFocusRegionSlices, numSlices, device)

				perceptLossStage2 = contentLoss.get_loss(inFocus, inFocusOutput)
				simiLossStage2 = losses.similarityLoss(inFocus, inFocusOutput, alpha, winSize)
				scaleLossStage2 += perceptLossW * perceptLossStage2 + simiLossW * simiLossStage2

				stepLossStage1 += scaleLossStage1
				stepLossStage2 += scaleLossStage2

				loss += stepLossStage1 + stepLossStage2
			
			epochLoss[0] += loss

			if jointTraining == True :
				optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters()), lr=lr,\
				betas=(beta1, beta2), eps=1e-08, weight_decay=0, amsgrad=False)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			else :
				stage1Parameters = list(model1.parameters()) + list(model2.parameters())
				stage2Parameters = list(model3.parameters())

				optimizerStage1 = optim.Adam(stage1Parameters, lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=0, amsgrad=False)
				optimizerStage2 = optim.Adam(stage2Parameters, lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=0, amsgrad=False)

				optimizerStage1.zero_grad()
				stepLossStage1.backward(retain_graph = True)
				optimizerStage1.step()

				optimizerStage2.zero_grad()
				stepLossStage2.backward()
				optimizerStage2.step()

			if step % 500 == 0 :
					print ("Step num :" + str(step) + " Train Loss :", str(loss.item()))

		with torch.no_grad() :
			validRealStep = 0
			for validStep, validSample in enumerate(validLoaderLF) :

				if validRealStep % 3 == 1 and validRealStep < numRealSteps:
					validRealStepLoss = torch.zeros(1).to(device)
					validRealInData = validLoaderRealList[realStep]['Input'].to(device)
					validRealSegMask = validLoaderRealList[realStep]['Mask'].to(device)
					validRealBlurOut = model1(validRealInData, validRealSegMask)
					for validRealDispIndex in range(len(validRealBlurOut)) :
						validRealScaleLoss = torch.zeros(1).to(device)
						validRealDispOut = validRealBlurOut[validRealDispIndex]
						validRealInFocus = F.interpolate(validRealInData, size = None, scale_factor = 1/pow(2.0, validRealDispIndex), mode = 'bilinear')
						validRealBlurMapSlices = helper.BlurMapsToSlices(validRealDispOut, numSlices, 6)
						validRealInFocusRegionSlices = []
						for validRealSlices in range(numSlices) :
							validRealNearFocusOutSlice = model2(validRealInFocus, validRealBlurMapSlices[validRealSlices])
							validRealBinaryBlurMapSlice = model3(validRealNearFocusOutSlice)
							validRealBinaryBlurMapSlice = torch.cat((validRealBinaryBlurMapSlice, validRealBinaryBlurMapSlice, validRealBinaryBlurMapSlice), dim = 1)
							validRealInFocusRegionSlice = torch.mul(validRealNearFocusOutSlice, validRealBinaryBlurMapSlice)
							validRealInFocusRegionSlices.append(validRealInFocusRegionSlice)
						validRealInFocusOutput = helper.CombineFocusRegionsToAIF(validRealInFocusRegionSlices, numSlices, device)
						validRealPerceptLoss = contentLoss.get_loss(validRealInFocus, validRealInFocusOutput)
						validRealSimiLoss = losses.similarityLoss(validRealInFocus, validRealInFocusOutput, alpha, winSize)
						validRealScaleLoss += perceptLossW * validRealPerceptLoss + simiLossW * validRealSimiLoss
						validRealStepLoss += validRealScaleLoss

					validRealStep += 1
					validLoss += validRealStepLoss

				validStepLossStage1 = torch.zeros(1).to(device)
				validStepLossStage1.requires_grad = False

				validStepLossStage2 = torch.zeros(1).to(device)
				validStepLossStage2.requires_grad = False

				validLoss = torch.zeros(1).to(device)
				validLoss.requires_grad = False
			
				validInData 	= validSample['Input'].to(device)
				validOutData = validSample['Output'].to(device)
				validInSegMask = validSample['Mask'].to(device)

				validBlurOut = model1(validInData, validInSegMask)


				for validDispIndex in range(len(validBlurOut)) :

					validScaleLossStage1 = torch.zeros(1).to(device)
					validScaleLossStage1.requires_grad = False

					validScaleLossStage2 = torch.zeros(1).to(device)
					validScaleLossStage2.requires_grad = False

					validDispOut 	= validBlurOut[validDispIndex]

					validInFocus	= F.interpolate(validInData, size = None, scale_factor = 1 / pow(2.0, validDispIndex), mode = 'bilinear')
					validNearFocusStackGT 	= F.interpolate(validOutData, size = None, scale_factor = 1 / pow(2.0, validDispIndex) , mode = 'bilinear')

					validNearFocusGTSlices = helper.FocalStackToSlices(validNearFocusStackGT, numSlices)
					validBlurMapSlices = helper.BlurMapsToSlices(validDispOut, numSlices, 6)
					validInFocusRegionSlices = []

				
					for validSlices in range(numSlices) :

						validSliceLoss = torch.zeros(1).to(device)
						validSliceLoss.requires_grad = False

						validNearFocusOutSlice = model2(validInFocus, validBlurMapSlices[validSlices])

						validPerceptLoss = contentLoss.get_loss(validNearFocusGTSlices[slices], validNearFocusOutSlice)
						validSimiLoss = losses.similarityLoss(validNearFocusGTSlices[slices], validNearFocusOutSlice, alpha, winSize)

						validSliceLoss = perceptLossW * validPerceptLoss + validSimiLoss * validSimiLoss

						validBinaryBlurMapSlice = model3(validNearFocusOutSlice)
						validBinaryBlurMapSlice = torch.cat((validBinaryBlurMapSlice, validBinaryBlurMapSlice, validBinaryBlurMapSlice), dim = 1)
						validBinaryLoss = losses.binarizationLoss(validBinaryBlurMapSlice) * binarizationLossW
						validSliceLoss += validBinaryLoss
						validInFocusRegionSlice = torch.mul(validNearFocusOutSlice, validBinaryBlurMapSlice)
						validInFocusRegionSlices.append(validInFocusRegionSlice)

						validScaleLossStage1 += validSliceLoss

					validInFocusOutput = helper.CombineFocusRegionsToAIF(validInFocusRegionSlices, numSlices, device)

					validPerceptLossStage2 = contentLoss.get_loss(validInFocus, validInFocusOutput)
					validSimiLossStage2 = losses.similarityLoss(validInFocus, validInFocusOutput, alpha, winSize)
					validScaleLossStage2 += perceptLossW * validPerceptLossStage2 + simiLossW * validSimiLossStage2

					validStepLossStage1 += validScaleLossStage1
					validStepLossStage2 += validScaleLossStage2

					validLoss += validStepLossStage1 + validStepLossStage2
			
				epochLoss[1] += validLoss

		epochLoss[0] /= step
		epochLoss[1] /= validStep
		epochFreq = epochSaveFreq
		print ("Time taken for epoch " + str(epoch) + " is " + str(time.clock() - start))
		
		if validLoss < minValidLoss :
			minValidLoss = validLoss
			minValidEpoch = epoch
			torch.save(model1.state_dict(), model1SaveDir + str(epoch) + extension)
			torch.save(model2.state_dict(), model2SaveDir + str(epoch) + extension)
			torch.save(model3.state_dict(), model3SaveDir + str(epoch) + extension)
			print ("Saving latest model at epoch: " + str(epoch))

		if epochCount % epochFreq == 0 :
			torch.save(model1.state_dict(), model1SaveDir + str(epoch) + extension)
			torch.save(model2.state_dict(), model2SaveDir + str(epoch) + extension)
			torch.save(model3.state_dict(), model3SaveDir + str(epoch) + extension)
					
		print ("Training loss for epoch " + str(epochCount) + " : " + str(epochLoss[0]))
		print ("Validation loss for epoch " + str(epochCount) + " : " + str(epochLoss[1]))

		trainLossArray[epoch] = epochLoss[0]
		validLossArray[epoch] = epochLoss[1]
	
	print (minValidEpoch)
	return (trainLossArray, validLossArray)

epochs 			= 200
batchSize 		= 2
epochSaveFreq	= 3
device 			= torch.device("cuda:0")
alpha			= 0.85
winSize			= 3
simiLossW	= 1.0
smoothLossW		= 0.0
perceptLossW = 0.1
binarizationLossW = 0.0
learningRate	= 0.0004
jointTraining = False
beta1			= 0.9
beta2			= 0.999

rootDirTrainLF    = '/home/sajal.maheshwari/BlurMapExp/FSDataLarge/trainData/'
rootDirValidLF    = '/home/sajal.maheshwari/BlurMapExp/FSDataLarge/validData/'
rootDirSave     = '/home/sajal.maheshwari/BlurMapExp/Baseline2SlicesInterleaved/checkpoints/'
rootDirTrainReal = '/home/sajal.maheshwari/BlurMapExp/SelfieDataAIF/trainData/'
rootDirValidReal = '/home/sajal.maheshwari/BlurMapExp/SelfieDataAIF/validData/'
	
#rootDirTrain 	= '/home/tirth.maniar/sajal/BlurMap/data/trainData/'
#rootDirValid 	= '/home/tirth.maniar/sajal/BlurMap/data/validData/'
#rootDirSave 	= '/home/tirth.maniar/sajal/BlurMap/checkpoints/'
#/home/tirth.maniar/sajal/BlurMap/data/trainData/

(trainLossArray, validLossArray) = train(epochs)
#trainLossArray = train(epochs)
#print (type(trainLossArray))
#print (type(validLossArray))
lossGraph = plt.plot(trainLossArray)
lossGraph = plt.plot(validLossArray)
#plt.savefig('/home/sajal.maheshwari/BlurMapExp/BlurMap/results/lossGraph.png')
