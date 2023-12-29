function [X, Y, label] = load_face_recog_dataset_with_noise(dataname, mode, data_normalization, variance)
data_name = strcat(dataname,'_',mode,'.mat');
load(data_name);

if(strcmp(mode,'wavelet'))
    X = X_origin;
    Y = X_wavelet;
end

if(strcmp(mode,'LBP'))
    X = X_origin;
    Y = X_LBP_regular;
end

if(strcmp(mode,'Downsample'))
    X = X_origin;
    Y = X_downsample;
end

% X = add_gw_noise(X, variance); 
% Y = add_gw_noise(Y, variance);

X = add_noise(X, variance); 
Y = add_noise(Y, variance);

X = double(X); Y = double(Y);

if(data_normalization == 1)
    X = my_normalize_tensor(X); Y = my_normalize_tensor(Y);
else
    X = X/max(X(:));  Y = Y/max(Y(:));
end

X = double(X); Y = double(Y);


end