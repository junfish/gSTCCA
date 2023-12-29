function [X_noisy] = add_gw_noise(X, variance)
%ADD_NOISE 此处显示有关此函数的摘要
%   此处显示详细说明


N_samples = size(X,ndims(X));
X_noisy = zeros(size(X));

for i = 1:N_samples
    Xi = normalize_X(X(:,:,i));
    Xi_noisy = imnoise(Xi,'gaussian',0,variance);
    X_noisy(:,:,i) = Xi_noisy;
end

end

function X_normalize = normalize_X(X)

max_X = max(X(:));
min_X = min(X(:));

X_normalize = (X - min_X)/(max_X - min_X);


end
