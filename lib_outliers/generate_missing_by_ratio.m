function [X_missing] = generate_missing_by_ratio(X, sampling_ratio)
%ADD_NOISE 此处显示有关此函数的摘要
%   此处显示详细说明

[H,W,N_samples] = size(X);
X_missing = X;

for i = 1:N_samples
    Xi = X(:,:,i);
    nnz_idx = find(Xi ~= 0);
    num_sample = floor(length(nnz_idx) * sampling_ratio);
    rng(i);idx_perm = randperm(length(nnz_idx));
    idx_selected = idx_perm(1:num_sample);
    nnz_idx_selected = nnz_idx(idx_selected);
    Xi(nnz_idx_selected) = 0;
    X_missing(:,:,i) = Xi;
end


end

function X_normalize = normalize_X(X)

max_X = max(X(:));
min_X = min(X(:));

X_normalize = (X - min_X)/(max_X - min_X);


end
