import torch
import torchvision
import pytorch_ssim
import matplotlib.pyplot as plt
import dataset as DataUtils
import numpy as np
import PIL.Image as Image
from torchvision import transforms


fmt = '.png'

gtDir = '/home/sajal/Desktop/BlurMapExperiment/Baseline/nonAppend/results/GroundTruth/'

baselinedir = '/home/sajal/Desktop/BlurMapExperiment/Baseline/nonAppend/results/OutputInFocus/'
modelDir = '/home/sajal/Desktop/BlurMapExperiment/BlurMapv9/results/OutputInFocus/'

imTransform = torchvision.transforms.ToTensor()

gtImages = DataUtils.make_dataset(gtDir, fmt)
baseImages = DataUtils.make_dataset(baselinedir, fmt)
modelImages = DataUtils.make_dataset(modelDir, fmt)

SSIMBaseArray = np.zeros(len(gtImages))
SSIMModelArray = np.zeros(len(gtImages))

diffArray = np.zeros(len(gtImages))
count = 0
#outLierList = []

for index in range (len(gtImages)) :
	
	gtImage = Image.open(gtImages[index])
	#imgplot = plt.imshow(gtImage)
	#plt.show()
	gtImage = imTransform(gtImage)
	gtImage = gtImage.view([1, 3, 256, 256])

	baseImage = Image.open(baseImages[index])
	#imgplot = plt.imshow(baseImage)
	#plt.show()
	baseImage = imTransform(baseImage)
	baseImage = baseImage.view([1, 3, 256, 256])


	modelImage = Image.open(modelImages[index])
	#imgplot = plt.imshow(modelImage)
	#plt.show()
	modelImage = imTransform(modelImage)
	modelImage = modelImage.view([1, 3, 256, 256])



	SSIMBaseArray[index] = pytorch_ssim.ssim(baseImage, gtImage, window_size = 3, size_average = True)
	SSIMModelArray[index] = pytorch_ssim.ssim(modelImage, gtImage, window_size = 3, size_average = True)
	diffArray[index] = SSIMModelArray[index] - SSIMBaseArray[index]
	if diffArray[index] >= 0 :
		count += 1
	else :
		print (index)
		print (diffArray[index])
print (count)

print (np.mean(SSIMBaseArray))
print (np.mean(SSIMModelArray))

print (np.std(SSIMBaseArray))
print (np.std(SSIMModelArray))

SSIMGraph = plt.plot(SSIMBaseArray)
SSIMGraph = plt.plot(SSIMModelArray)
plt.savefig('/home/sajal/Desktop/BlurMapExperiment/BlurMapv9/results/SSIMGraph.png')

plt.figure()
diffGraph = plt.plot(diffArray)
plt.savefig('/home/sajal/Desktop/BlurMapExperiment/BlurMapv9/results/diffGraph.png')