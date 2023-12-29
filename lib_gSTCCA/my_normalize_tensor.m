function [X mu] = my_normalize_tensor(X)
%NORMALIZE Normalize the columns (variables) of a data matrix to unit
%Euclidean length.
%
%   X = NORMALIZE(X) centers and scales the observations of a data matrix
%   such that each variable (column) has unit Euclidean length. For a
%   normalized matrix X, X'*X is equivalent to the correlation matrix of X.
%
%   [X MU] = NORMALIZE(X) also returns a vector MU of mean values for each
%   variable. 
%
%   [X MU D] = NORMALIZE(X) also returns a vector D containing the
%   Euclidean lengths for each original variable.
%
%   This function is an auxiliary part of SpAM, a matlab toolbox for
%   sparse modeling and analysis.
%
%  See also CENTER.

size_X = size(X);
ndim_X = ndims(X);

X = my_tenmat(X,ndims(X));

n = size(X,1);

[Xt mu] = center(X);
s = std(X,1,1)';
s(s==0) = 1;
X = Xt./(repmat(s',n,1));

X = mat_ten(X,ndim_X,size_X);
