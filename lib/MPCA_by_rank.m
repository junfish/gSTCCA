function [U] = MPCA_by_rank(X, r)
%MPCA 此处显示有关此函数的摘要
%   此处显示详细说明
dim_X = ndims(X);X = tensor(X);
for i = 1:dim_X-1
    Xi = double(tenmat(X,i));
    Ui = cal_U(Xi,r);
    U{i} = Ui;
end

end

function U = cal_U(X,rank)
[U,S,~] = svd(X*X');
U = U(:,1:rank);

end