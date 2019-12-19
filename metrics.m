clear

commonPath = '/home/sajal/Desktop/BufferResutlsFinal/';
%/home/sajal/Desktop/BufferResutlsFinal/Baseline2SlicesInterleaved/results/SliceOutputs/outputSlice1/GroundTruth
%refPath = 'SliceOutputs/outputSlice1/FocalSlices/';
%resultPath = 'SliceOutputs/outputSlice10/FocalSlices/';
%refPath = 'InputAIF/';
%resultPath = 'SliceOutputs/outputSlice10/FocalSlices/';
%refPath = 'InputAIF/';
%resultPath = 'SliceOutputs/outputSlice1/FocalSlices/';
refPath = 'Baseline2SlicesInterleaved/results/SliceOutputs/outputSlice10/GroundTruth/';
resultPath = 'SiggAsiaResults/results/outputSlice10/';
numImages = 501;
psnrScore = 0;
ssimScore = 0;

for i=1:1:numImages
    refIm = imread(strcat(commonPath, refPath, num2str(i-1), '.png'));
    resultIm = imread(strcat(commonPath, resultPath, num2str(i-1), '.png'));
    ssimScore = ssimScore + ssim(refIm, resultIm);
    psnrScore = psnrScore + psnr(refIm, resultIm);
end

ssimScore / numImages
psnrScore / numImages
