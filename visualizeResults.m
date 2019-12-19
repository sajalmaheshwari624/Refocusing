clear
clc

commonPath = '/home/sajal.maheshwari/BlurMapExp/Baseline2SlicesInterleaved/results/';

input = 'InputAIF/';
masks = 'GroundTruthMasks/';
outputSlice1 = 'SliceOutputs/outputSlice1/FocalSlices/';
outputSlice10 = 'SliceOutputs/outputSlice10/FocalSlices/';
mapSlice1 = 'SliceOutputs/outputSlice1/BinaryMaps/';
mapSlice10 = 'SliceOutputs/outputSlice10/BinaryMaps/';
output = 'OutputAIF/';

outputFolder = 'visualizerLF/';
numImages = 501;

for i=1:numImages
    i
    close all
    inputIm = imread(strcat(commonPath, input, num2str(i-1), '.png'));
    maskIm = imread(strcat(commonPath, masks, num2str(i-1), '.png'));
    outputSlice1Im = imread(strcat(commonPath, outputSlice1, num2str(i-1), '.png'));
    outputSlice10Im = imread(strcat(commonPath, outputSlice10, num2str(i-1), '.png'));
    mapSlice1Im = imread(strcat(commonPath, mapSlice1, num2str(i-1), '.png'));
    mapSlice10Im = imread(strcat(commonPath, mapSlice10, num2str(i-1), '.png'));
    outputIm = imread(strcat(commonPath, output, num2str(i-1), '.png'));
    
    catIm = cat(2, inputIm, maskIm, outputSlice1Im, mapSlice1Im, outputSlice10Im, mapSlice10Im, outputIm);
    imwrite(catIm, strcat(commonPath, outputFolder, num2str(i-1), '.png'));
end