clear
clc

maxValue = 0;
minValue = 255;
for i=0:343
    a = imread(strcat('/home/tirth.maniar/sajal/BlurMap/results/OutputMaps/', num2str(i), '.png'));
    b(:,:,1) = imadjust(a(:,:,1));
    b(:,:,2) = imadjust(a(:,:,2));
    b(:,:,3) = imadjust(a(:,:,3));
    mina = min(a(:));
    maxa = max(a(:));
    if mina < minValue
        minValue = mina
        i
    end
    if maxa > maxValue
        maxValue = maxa
        i
    end
    imwrite(b, strcat('/home/tirth.maniar/sajal/BlurMap/results/OutputMaps/b', num2str(i), '.png'));
end