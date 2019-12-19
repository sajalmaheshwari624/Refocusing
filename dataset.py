import os
import os.path
import torch
import torchvision
import torch.utils.data as data
import PIL.Image as Image

def is_image_file(filename, IMG_Extension):
		return any(filename.endswith(extension) for extension in IMG_Extension)

def make_dataset(dir, IMG_Extension):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname, IMG_Extension):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class Data(data.Dataset) :
	def __init__(self, transforms, root_dir, output_dirs, input_dir, segm_dir) :
		self.root_dir = root_dir
		self.input_dir = os.path.join(root_dir, input_dir)
		self.segm_dir = os.path.join(root_dir, segm_dir)
		self.output_dir = []
		self.numSlices = len(output_dirs)
		for index in range(self.numSlices) :
			self.output_dir.append(os.path.join(root_dir, output_dirs[index]))
		self.transforms = transforms
		self.IMG_Extension = ['.png']
		self.inImages = make_dataset(os.path.join(self.root_dir, self.input_dir), self.IMG_Extension)
		self.segMasks = make_dataset(os.path.join(self.root_dir, self.segm_dir), self.IMG_Extension)
		self.outImages = []
		for index in range(self.numSlices) :
			self.outImages.append(make_dataset(os.path.join(self.root_dir, self.output_dir[index]), self.IMG_Extension))

	def __getitem__(self, index) :
		inImage = Image.open(self.inImages[index])
		inImage = self.transforms(inImage)
		segMask = Image.open(self.segMasks[index])
		segMask = self.transforms(segMask)
		#inImage = inImage.mean(0)
		#inImage = inImage.view([1, 256, 256])
		for i in range(self.numSlices) :
			outImage = Image.open(self.outImages[i][index])
			outImage = self.transforms(outImage)
			if i == 0 :
				outTensor = outImage
			else :
				outTensor = torch.cat((outTensor, outImage), dim = 0)
		sample = {'Input': inImage, 'Output': outTensor, 'Mask' : segMask}
		return sample

	def __len__(self) :
		return len(self.inImages)

class InFocusData(data.Dataset) :
	def __init__(self, transforms, root_dir, input_dir, mask_dir) :
		self.root_dir = root_dir
		self.input_dir = os.path.join(root_dir, input_dir)
		self.mask_dir = os.path.join(root_dir, mask_dir)
		self.transforms = transforms
		self.IMG_Extension = ['.png']
		self.images = make_dataset(self.input_dir, self.IMG_Extension)
		self.masks = make_dataset(self.mask_dir, self.IMG_Extension)

	def __getitem__(self, index) :
		inImage = Image.open(self.images[index])
		inImage = self.transforms(inImage)
		segMask = Image.open(self.masks[index])
		segMask = self.transforms(segMask)
		sample = {'Input': inImage, 'Mask' : segMask}
		return sample

	def __len__(self) :
		return len(self.images)

#device = torch.device('cpu')
#imTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#trainData = Data(imTransform, root_dir = '/home/sajal/Desktop/BlurMap/data/trainData/', input_dir = 'input', output_dir = 'output')
#trainLoader = torch.utils.data.DataLoader(trainData, batch_size=2, shuffle=True)
