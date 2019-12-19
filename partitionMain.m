clear
clc
close all

rng(3)
fmt = '.png';

commonPath      = '/home/sajal.maheshwari/BlurMapExp/BlurMap/data/';
inPathinput     = 'InFocusLF/';
inPathoutput    = 'NearFocusLF/';

outFolders = {'trainData/'; 'validData/'; 'testData/'};

trainPartition  = 0.868;
validPartition  = 0.029;
testPartition   = 0.103;

numImages   = 3343;
randomOrder = randperm(numImages);

trainSet    = randomOrder(1:round(trainPartition * numImages));
validSet    = randomOrder(numel(trainSet) + 1 : round((trainPartition + validPartition) * numImages));
testSet     = randomOrder(numel(trainSet) + numel(validSet) + 1 : numImages);

setCells = {trainSet, validSet, testSet};

for i=1:1:numel(outFolders)
    success = partitionData(strcat(commonPath, inPathinput), ...
        strcat(commonPath, cell2mat(outFolders(i)), 'input/'), cell2mat(setCells(i)), fmt);
   success = partitionData(strcat(commonPath, inPathoutput), ...
       strcat(commonPath, cell2mat(outFolders(i)), 'output/'), cell2mat(setCells(i)), fmt);
end