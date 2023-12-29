function [Ux, Uy, corr, Mx, My] = NDCCA_youlin(X, Y, dim, max_iters)

% Remove mean
dim_x = ndims(X);
dim_y = ndims(Y);

Mx=mean(X,dim_x);
My=mean(Y,dim_y);
X=bsxfun(@minus,X,Mx);
Y=bsxfun(@minus,Y,My);

size_X = size(X);
size_Y = size(Y);

Nx = size_X(dim_x);
Ny = size_Y(dim_y);

if(Nx ~= Ny)
    disp('wrong number of samples');
    return
end

if(dim == 1)
    method = 'one_step';
else
    method = 'Deflation';
end

Ux = initialize_U(size_X, dim);
Uy = initialize_U(size_Y, dim);

for iter = 1:max_iters
    if(dim == 1)
        [Ux, Uy] = update_U_one_dim(X,Y,Ux,Uy);
    else
        [Ux, Uy] = update_U_deflation(X,Y,Ux,Uy,dim);
    end
end

if(dim == 1)
    corr = cal_corr(X,Y,Ux,Uy);
else
    corr = 0;
end

end

function U = initialize_U(size_U, dim)

dim_data = length(size_U); U  = cell(dim_data,1);

for i = 1:dim_data
    size_i = size_U(i);
%     rng('default');U{i} = randn(size_i, dim); U{i} = U{i}/norm(U{i});
    U{i} = randn(size_i, dim); U{i} = U{i}/norm(U{i});
    if(i == dim_data)
        U{i} = eye(100, 100);
    end
end

end

function [Ux, Uy] = update_U_one_dim(X,Y,Ux,Uy)

dim_data = ndims(X);

for mode_update = 1:dim_data - 1
    [U_mode_x_new, U_mode_y_new] = update_U_oneStep(X,Y,Ux,Uy,mode_update);
    Ux{mode_update} = U_mode_x_new;
    Uy{mode_update} = U_mode_y_new;
end

end

function [U_mode_x, U_mode_y] = update_U_oneStep(X,Y,Ux,Uy,mode_update)

dim_data = ndims(X);

mode_all = 1:dim_data-1; proj_mode = setdiff(mode_all,mode_update);
N = size(double(X),ndims(double(X)));

X_proj = ttm(tensor(X),Ux,proj_mode,'t');
Y_proj = ttm(tensor(Y),Uy,proj_mode,'t');

X_mode = squeeze(X_proj); Y_mode = squeeze(Y_proj);

X_mode = double(X_mode); Y_mode = double(Y_mode);

U_mode_x = Ux{mode_update};U_mode_y = Uy{mode_update};

[U_mode_x, U_mode_y] = solve_CCA_youlin(X_mode,Y_mode,U_mode_x,U_mode_y,N);

end

function [Ux, Uy] = update_U_deflation(X,Y,Ux,Uy,rank)

dim_data = ndims(X);

for k = 1:rank
    [X_res, Y_res] = cal_res(X, Y, Ux, Uy, k, rank);
    % Update Uk and Vk iteratively
    Ux_k = extract_U(Ux, k); Uy_k = extract_U(Uy, k);
    for t = 1:10
        [Ux_k, Uy_k] = update_U_one_dim(X_res,Y_res,Ux_k,Uy_k);
    end
    
    for mode = 1:dim_data - 1
         Ux{mode}(:,k) = Ux_k{mode};
         Uy{mode}(:,k) = Uy_k{mode};
    end
    
end

end

function [U_mode_x, U_mode_y] = update_U_oneStep_for_deflation(X,Y,Ux,Uy,U_mode_x,U_mode_y,mode)

dim_data = ndims(X);

mode_all = 1:dim_data-1; proj_mode = setdiff(mode_all,mode);
N = size(double(X),ndims(double(X)));

X_proj = ttm(tensor(X),Ux,proj_mode,'t');
Y_proj = ttm(tensor(Y),Uy,proj_mode,'t');

X_mode = tenmat(X_proj, mode);
Y_mode = tenmat(Y_proj, mode);

X_mode = double(X_mode); Y_mode = double(Y_mode);

[U_mode_x, U_mode_y] = solve_CCA_youlin(X_mode,Y_mode,U_mode_x,U_mode_y,N);

end

function [U_mode_x,U_mode_y] = solve_CCA_youlin(X_mean,Y_mean,U_mode_x,U_mode_y,N)

[Fx, ~] = size(X_mean); [Fy, ~] = size(Y_mean);

Sxx = X_mean * X_mean';
Syy = Y_mean * Y_mean';
Sxy = (1.0/N)*(X_mean * Y_mean');

Sxx = (1.0/N)*Sxx +(1e-4) * eye(Fx); % regularized CCA

Syy = (1.0/N)*Syy + (1e-4) * eye(Fy); % regularized CCA

% Update Lx and Ly
U_mode_x = Sxx^(-1) * Sxy * U_mode_y;
U_mode_x = U_mode_x/norm(U_mode_x);

U_mode_y = Syy^(-1) * Sxy' * U_mode_x;
U_mode_y = U_mode_y/norm(U_mode_y);

end

function corr = cal_corr(X,Y,Ux,Uy)
dim_data = ndims(X);
mode_proj = 1:dim_data - 1;

X_proj = ttm(tensor(X), Ux, mode_proj, 't');
Y_proj = ttm(tensor(Y), Uy, mode_proj, 't');

X_proj = double(X_proj); Y_proj = double(Y_proj); 
X_proj = squeeze(X_proj); Y_proj = squeeze(Y_proj);

corr = X_proj' * Y_proj/(sqrt(X_proj'*X_proj) * sqrt(Y_proj'*Y_proj));

end

function [X_res, Y_res] = cal_res(X, Y, Ux, Uy, k, rank)

dim_data = ndims(X); N = size(X,dim_data);
mode_proj = 1:dim_data-1;
idx_all = 1:rank;
idx_res = setdiff(idx_all,k);
Tk = zeros(N, rank-1); Sk = zeros(N, rank-1);

for i = 1:rank-1
    r = idx_res(i);
    
    Ux_r = extract_U(Ux,r);
    col_i = ttm(tensor(X),Ux_r,mode_proj,'t');
    Tk(:,i) = double(col_i);
    
    Uy_r = extract_U(Uy,r);
    col_i = ttm(tensor(Y),Uy_r,mode_proj,'t');
    Sk(:,i) = double(col_i);
end

T = Tk*Tk'; It = eye(size(T)); T_res = It - T;
X_res = ttm(tensor(X),T_res,dim_data);

S = Sk*Sk'; Is = eye(size(S)); S_res = Is - S;
Y_res = ttm(tensor(Y),S_res,dim_data);

X_res = double(X_res); Y_res = double(Y_res);


end

function Ur = extract_U(U,r)

dim_data = length(U); Ur = cell(dim_data,1);

for i = 1:dim_data
    Ui = U{i};
    Ur_i = Ui(:,r);
    Ur{i} = Ur_i;
    if(i == dim_data)
        Ur{i} = Ui;
    end
end

end
