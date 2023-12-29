function [LBP] = extract_LBP_feature(Input)
%EXTRACT_DWT2_FEATURE 此处显示有关此函数的摘要

imgSize = size(Input);
if numel(imgSize) > 2
    imgG = rgb2gray(Input);
else
    imgG = Input;
end
[rows, cols] = size(imgG);
rows=int16(rows);
cols=int16(cols);
LBP = uint8(zeros(rows-2, cols-2));

for i=2:rows-2
    for j=2:cols-2
        center = imgG(i,j);
        lbpCode = 0;
        lbpCode = bitor(lbpCode, (bitshift(compareCenter(i-1, j-1, center, imgG), 7)));
        lbpCode = bitor(lbpCode, (bitshift(compareCenter(i-1,j, center, imgG), 6)));
        lbpCode = bitor(lbpCode, (bitshift(compareCenter(i-1,j+1, center, imgG), 5)));
        lbpCode = bitor(lbpCode, (bitshift(compareCenter(i,j+1, center, imgG), 4)));
        lbpCode = bitor(lbpCode, (bitshift(compareCenter(i+1,j+1, center, imgG), 3)));
        lbpCode = bitor(lbpCode, (bitshift(compareCenter(i+1,j, center, imgG), 2)));
        lbpCode = bitor(lbpCode, (bitshift(compareCenter(i+1,j-1, center, imgG), 1)));
        lbpCode = bitor(lbpCode, (bitshift(compareCenter(i, j-1, center, imgG), 0)));
        LBP(i-1,j-1) = lbpCode;
    end
end

LBP = imresize(LBP, size(Input));

end

function flag = compareCenter(x, y, center, imgG)
if imgG(x, y) > center
    flag = 1;
else
    flag = 0;
end
end


