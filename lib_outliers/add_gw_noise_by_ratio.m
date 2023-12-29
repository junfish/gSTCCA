function [X_noisy] = add_gw_noise_by_ratio(X, ratio, variance)
%ADD_NOISE 此处显示有关此函数的摘要
%   此处显示详细说明

rng(666);
N_samples = size(X,ndims(X));
N_selected = floor(N_samples * ratio);
rng_idx = randperm(N_selected);
idx_selected = rng_idx(1:N_selected);
X_noisy = X;


for i = 1:N_selected
    idx_i = idx_selected(i);
    Xi = X(:,:,idx_i);
    Xi = double(Xi)/255;
%     Xi = normalize_X(Xi);
    Xi_noisy = imnoise(Xi,'gaussian',0,variance);
    X_noisy(:,:,idx_i) = Xi_noisy;
end

% for i = 1:N_samples
%     Xi = X(:,:,i);
%     Xi = normalize_X(Xi);
%     Xi_noisy = imnoise(Xi,'gaussian',0,variance);
%     X_noisy(:,:,i) = Xi_noisy;
% end

end

function X_normalize = normalize_X(X)

max_X = max(X(:));
min_X = min(X(:));

X_normalize = (X - min_X)/(max_X - min_X);


end
