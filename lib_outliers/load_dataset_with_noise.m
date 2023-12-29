function [X,Y,label] = load_dataset_with_noise(dataset_name, data_normalization, ratio, noise_variance)
%LOAD_DATASET 此处显示有关此函数的摘要
%   此处显示详细说明

load(dataset_name);
if(strcmp(dataset_name, 'Gait17_32x22x10_processed') == 1)
    X = double(U); Y = double(V);
end

if(strcmp(dataset_name, 'Mnist_012_processed_new') == 1)
    X = upper_all; Y = lower_all;
    X = add_gw_noise_by_ratio(X, ratio, noise_variance);
    Y = add_gw_noise_by_ratio(Y, ratio, noise_variance);
end

if(strcmp(dataset_name,'HIV') == 1)
    X = dti; Y = fmri;
end

if(strcmp(dataset_name,'BP') == 1)
    X = dti; Y = fmri;
end

if(strcmp(dataset_name,'PPMI_new') == 1)
    data = X_normalize;
    X = data(:,:,1:2,:); Y = data(:,:,3,:);
    label = label_selected;
end

if(strcmp(dataset_name,'PPMI_new_balance') == 1)
    data = X;
    X = data(:,:,3,:); Y = data(:,:,1:2,:);
    X = squeeze(X); Y = (Y(:,:,1,:) + Y(:,:,2,:))/2; Y = squeeze(Y);
    label = label_new;
end


if(data_normalization == 1)
    X = my_normalize_tensor(X); Y = my_normalize_tensor(Y);
    %     X = normalize_to_zero_one(X); Y = normalize_to_zero_one(Y);
else
%     X = X/max(X(:));  Y = Y/max(Y(:));
end


end

