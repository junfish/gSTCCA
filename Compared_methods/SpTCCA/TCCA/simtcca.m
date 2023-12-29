function [X, Y] = simtcca(V, W, rho, n, varargin)
% SIMCCA Simulate joint random normal deviates with the requested canonical
% tensors (V, W) in CP format and the ceoff rho assuming marginal separable
% covariances in X and Y
%
% INPUT:
%   V: p1-by-p2-by-...-p_{d_x} left canonical tensor
%   W: q1-by-q2-by-...-q_{d_y} right canonical tensor
%   rho: canonical correlation between 0 and 1
%   n: number of replicates to be simulated
%
% OPTIONAL NAME-VALUE PAIRs:
%   mux: mean of X random tensors
%   muy: mean of Y random tensors
%   noisex: noise level in X tensors
%   noisey: noise level in Y tensors
%
% OUTPUT:
%   X: (p1,...,p_{d_x})-by-n simulated x random tensors
%   Y: (q1,...,q_{d_y})-by-n simulated y random tensors

%% parse inputs
argin = inputParser;
argin.addRequired('V', @(x) isa(x,'ktensor') || isnumeric(x));
argin.addRequired('W', @(x) isa(x,'ktensor') || isnumeric(x));
argin.addRequired('rho', @isnumeric);
argin.addRequired('n', @isnumeric);
argin.addParamValue('mux', zeros(prod(size(V)), 1), @isnumeric); %#ok<*PSIZE>
argin.addParamValue('muy', zeros(prod(size(V)), 1), @isnumeric);
argin.addParamValue('noisex', 1e-3, @(x) x>=0);
argin.addParamValue('noisey', 1e-3, @(x) x>=0);
argin.parse(V, W, rho, n, varargin{:});
mux = argin.Results.mux;
muy = argin.Results.muy;
noisex = argin.Results.noisex;
noisey = argin.Results.noisey;

% retrieve dimension
px = size(V);
qy = size(W);
dx = ndims(V);
dy = ndims(W);
rx = size(V{1}, 2);
ry = size(W{1}, 2);

% standardize V.lambda and W.lambda to be 1
for i = 1:dx
  V.U{i} = bsxfun(@times, V.U{i}, (V.lambda').^(1/dx));
end
for j = 1:dy
  W.U{j} = bsxfun(@times, W.U{j}, (W.lambda').^(1/dy));
end

% set separable covariances for X
Sigmax = cell(dx, 1);
Cxx = 1;
cumkrx = ones(1, rx);
for i = 1:dx
  [Qxi, Rxi] = qr(V{i}, 0);
  Sigmax{i} = Qxi / Rxi';
  Sigmax{i} = ...
    Sigmax{i} * Sigmax{i}' / rx^(1/dx) + noisex * (eye(px(i)) - Qxi * Qxi');
  Cxx = kron(Sigmax{i}, Cxx);
  cumkrx = khatrirao(Sigmax{i} * V{i}, cumkrx);
end
% set separable covariances for Y
Sigmay = cell(dy, 1);
Cyy = 1;
cumkry = ones(1, ry);
for j = 1:dy
  [Qyj, Ryj] = qr(W{j}, 0);
  Sigmay{j} = Qyj / Ryj';
  Sigmay{j} = ...
    Sigmay{j} * Sigmay{j}' / ry^(1/dy) + noisey * (eye(qy(j)) - Qyj * Qyj');
  Cyy = kron(Sigmay{j}, Cyy);
  cumkry = khatrirao(Sigmay{j} * W{j}, cumkry);
end

% form the joint covariance matrix
Cxy = rho * sum(cumkrx, 2) * sum(cumkry, 2)';
C = [Cxx Cxy; Cxy' Cyy];

% % check whether we get the desired canonical coefficient
% A = [zeros(size(Cxx)) Cxy; Cxy' zeros(size(Cyy))];
% B = [Cxx zeros(size(Cxy)); zeros(size(Cxy')) Cyy];
% display(eigs(A, B, 6));

% simulate random normal deviates
RV = mvnrnd([mux' muy'], C, n);
X = permute(tensor(RV(:, 1:prod(px)), [n, px]), [2:dx+1, 1]);
Y = permute(tensor(RV(:, prod(px)+1:end), [n, qy]), [2:dy+1, 1]);
