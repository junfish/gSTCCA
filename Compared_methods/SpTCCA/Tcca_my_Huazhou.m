function [Ux, Uy, rho, Mx, My] = Tcca_my_Huazhou(X, Y, rx, ry, max_iters)
%TCCA_MY an implementation of TCCA of HuaZhou's paper.
%   此处显示详细说明

% Remove mean
dim_x = ndims(X);
dim_y = ndims(Y);

Mx=mean(X,dim_x);
My=mean(Y,dim_y);
X=bsxfun(@minus,X,Mx);
Y=bsxfun(@minus,Y,My);

% starting point
Ux = initialize_U(X, rx);
Uy = initialize_U(Y, ry);

for iter = 1:max_iters
    % mode of X
    mode_x = mod(iter-1, dim_x-1) + 1;
    % mode of Y
    mode_y = mod(iter-1, dim_y-1) + 1;
    
    [U_mode_x, U_mode_y] = update_U(X, Y, Ux, Uy, rx, ry, mode_x, mode_y);
    Ux{mode_x} = U_mode_x; 
    Uy{mode_y} = U_mode_y;  
end

canvarx = cal_canvar(X, Ux);
canvary = cal_canvar(Y, Uy);
rho = corr(canvarx, canvary);

end

function [Ux] = initialize_U(X, dim)
size_X = size(X); N = size_X(end);
dx = ndims(X) - 1;

betax = ktensor(arrayfun(@(j) randn(size_X(j),dim), 1:dx, ...
    'UniformOutput',false));
Ux = betax.U;Ux{dx+1} = eye(N);

% H = size_X(1); W = size_X(2); C = size_X(3); N = size_X(4);
% Ux{1} = randn(H, dim);Ux{1} = Ux{1}/norm(Ux{1},'fro');
% Ux{2} = randn(W, dim);Ux{2} = Ux{2}/norm(Ux{2},'fro');
% Ux{3} = randn(C, dim);Ux{3} = Ux{3}/norm(Ux{3},'fro');
% Ux{4} = ones(N, N);

end

function [U_mode_x, U_mode_y] = update_U(X, Y, Ux, Uy, rx, ry, mode_x, mode_y)
N = size(X,ndims(X));
dim_x = ndims(X); dim_y = ndims(Y);
size_X = size(X); size_Y = size(Y);
U_proj_x = cal_proj_U(Ux, mode_x);
U_proj_y = cal_proj_U(Uy, mode_y);

vec_x = cal_vec_Tensor(X, mode_x);
vec_y = cal_vec_Tensor(Y, mode_y);

Sx = U_proj_x'*vec_x;
Sxx = Sx * Sx'; Sxx = Sxx/N + (1e-3 * eye(size(Sxx)));

Sy = U_proj_y'*vec_y;
Syy = Sy * Sy'; Syy = Syy/N + (1e-3 * eye(size(Syy)));

Sxy = Sx * Sy'/N;

Sxx_inv = Sxx^(-0.5); Syy_inv = Syy^(-0.5);

T = Sxx_inv*Sxy*Syy_inv;
[U,D,V] = svd(T);
u_x = Sxx_inv*U(:,1);
u_y = Syy_inv*V(:,1);

if(dim_x == 2)
    U_mode_x = Sxx_inv * U(:,1:rx);
else  
    U_mode_x = reshape(u_x,[size_X(mode_x), rx]);
end

if(dim_y == 2)
    U_mode_y = Syy_inv * V(:,1:ry);
else  
    U_mode_y = reshape(u_y,[size_Y(mode_y), ry]);
end



end

function U_proj = cal_proj_U(U, mode_proj)

dim_x = length(U) - 1;
if(dim_x == 1)
    [p,r] = size(U{1});
    U_proj = eye(p);
else
    mode_all = 1:dim_x;
    mode_cal = setdiff(mode_all, mode_proj);
    mode_cal = sort(mode_cal,'descend');    
    U_proj = U{mode_cal(1)};
    mode_cal

    for i = 2:dim_x - 1
        mode_cal_i = mode_cal(i);
        U_proj = khatrirao(U_proj, U{mode_cal_i});
    end
    
    di = size(U{mode_proj},1);
    U_proj = kron(U_proj, eye(di));
end

end

function vec_T = cal_vec_Tensor(T, mode_t)

dim_T = ndims(T)-1;
vec_T = double(tenmat(tensor(T), [mode_t 1:mode_t-1 mode_t+1:dim_T], dim_T+1));

end

function canvar = cal_canvar(X, Ux)
[~,r] = size(Ux{1});dim_x = ndims(X);
mode_proj = 1:(dim_x - 1);

canvar = 0;
for i = 1:r
    Ui = fetch_U(Ux, i);
    canvar_i = ttm(tensor(X), Ui, mode_proj,'t');
    canvar_i = double(squeeze(canvar_i));
    canvar = canvar + canvar_i;
end

end

function Ur = fetch_U(U,r)

dim_x = length(U); Ur = cell(dim_x,1);
for i = 1:dim_x
    Ui = U{i};
    Ur_i = Ui(:,r);
    Ur{i} = Ur_i;
end

Ur{dim_x} = U{dim_x};

end

