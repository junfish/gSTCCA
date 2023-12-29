function [Ux, Uy, Mx, My] = NDCCA(X, Y, dim)

% Remove mean
dim_data = ndims(X);

Mx=mean(X,dim_data);
My=mean(Y,dim_data);
X=bsxfun(@minus,X,Mx);
Y=bsxfun(@minus,Y,My);

size_X = size(X);
size_Y = size(Y);

Nx = size_X(dim_data);
Ny = size_Y(dim_data);

if(Nx ~= Ny)
    disp('wrong number of samples');
    return
end

max_iters = 100;
Ux = initialize_U(size_X, dim);
Uy = initialize_U(size_Y, dim);

for iter = 1:max_iters
    for mode = 1:dim_data - 1
        [U_mode_x, U_mode_y, corr] = update_U(X,Y,Ux,Uy,dim,mode);
        Ux{mode} = U_mode_x;
        Uy{mode} = U_mode_y;
%         if(mode == 1)
%             corr
%         end
    end
end

end

function U = initialize_U(size_U, dim)

dim_data = length(size_U); U  = cell(dim_data,1);

for i = 1:dim_data
    size_i = size_U(i);
    rng('default');U{i} = randn(size_i, dim);
    if(i == dim_data)
        U{i} = eye(size_i, size_i);
    end
end

end

function [U_mode_x, U_mode_y, corr] = update_U(X,Y,Ux,Uy,dim,mode)

dim_data = ndims(X);

mode_all = 1:dim_data-1; proj_mode = setdiff(mode_all,mode);
N = size(double(X),ndims(double(X)));

X_proj = ttm(tensor(X),Ux,proj_mode,'t');
Y_proj = ttm(tensor(Y),Uy,proj_mode,'t');

% [U_mode_x, U_mode_y, corr] = solve_CCA(X_proj,Y_proj,mode,dim);

X_mode = tenmat(X_proj, mode);
Y_mode = tenmat(Y_proj, mode);

X_mode = double(X_mode);
Y_mode = double(Y_mode);

[U_mode_x, U_mode_y, corr] = solve_CCA_new(X_mode,Y_mode,dim, N);

end

function [A,B,corr] = solve_CCA_new(X_mean,Y_mean, dim, N)

[Fx, ~] = size(X_mean); [Fy, ~] = size(Y_mean);

r_dim = min(Fx, Fy);
r_dim = min(r_dim, dim);

Sxx = (X_mean * X_mean');
Syy = (Y_mean * Y_mean');
Sxy = (X_mean * Y_mean')/N;

if(Fx > N)
    Sxx = Sxx/N + 0.01 * eye(Fx); % regularized CCA
else
    Sxx = Sxx/N;
end

if(Fy > N)
    Syy = Syy/N + 0.01 * eye(Fy); % regularized CCA
else
    Syy = Syy/N;
end

[Vx,Dx] = eig((Sxx+Sxx')/2);
[Vy,Dy] = eig((Syy+Syy')/2);
Dx = diag(Dx);
Dy = diag(Dy);
Sxx_inv = Vx*diag(real(Dx.^(-1/2)))*Vx';
Syy_inv = Vy*diag(real(Dy.^(-1/2)))*Vy';

T = Sxx_inv*Sxy*Syy_inv;
[U,D,V] = svd(T);
r=diag(D);
A = Sxx_inv*U(:,1:r_dim);
B = Syy_inv*V(:,1:r_dim);
r = r(1:r_dim);
corr = sum(r);

end
