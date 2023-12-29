function [X,Y,label] = load_JAFFE_dataset(data_normalization, mode)
%LOAD_DATASET 此处显示有关此函数的摘要
%   此处显示详细说明
if(strcmp(mode,'paired') == 1)
    load jaffe_gabor_feature_pair
    X = Gabor_Feature_Tensor1; Y = Gabor_Feature_Tensor2;
end

if(strcmp(mode,'unpaired') == 1)
    load jaffe_gabor_feature_pair
    load img_cropped
    X = double(img_all); Y = Gabor_Feature_Tensor2;
end

if(data_normalization == 1)
    X = my_normalize_tensor(X); Y = my_normalize_tensor(Y);
else
    X = X/max(X(:));  Y = Y/max(Y(:));
end

label = label_all;


end

