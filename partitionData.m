function[success] = partitionData(inputPath, outputPath, partitionSet, fmt)
success = true;

for i = 1:1:numel(partitionSet)
    i
    im = imread(strcat(inputPath, num2str(partitionSet(i)), fmt));
    imwrite(im, strcat(outputPath, num2str(partitionSet(i)), fmt));
end
end