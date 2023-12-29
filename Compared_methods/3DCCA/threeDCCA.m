function [Ux, Uy, Mx, My] = threeDCCA(X, Y, dim)

% Remove mean
Mx=mean(X,4);
My=mean(Y,4);
X=bsxfun(@minus,X,Mx);
Y=bsxfun(@minus,Y,My);

[Hx,Wx,Cx,Nx] = size(X);
[Hy,Wy,Cy,Ny] = size(Y);

if(Nx ~= Ny)
    disp('wrong number of samples');
    return
end

N = Nx;


max_iters = 50;
Ux = initialize_U([Hx, Wx, Cx, N], dim);
Uy = initialize_U([Hy, Wy, Cy, N], dim);

corr_old = 0;
for iter = 1:max_iters
    corr_new = 0;
    for mode = 1:3
        [U_mode_x, U_mode_y, corr] = update_U(X,Y,Ux,Uy,dim,mode);
        Ux{mode} = U_mode_x;
        Uy{mode} = U_mode_y;
        corr_new = corr_new + corr;
    end
    
    if(abs(corr_new - corr_old) < 0.001)
        break
    else
        corr_old = corr_new;
    end
end

end

function U = initialize_U(size_U, dim)

U = cell(4,1);
H = size_U(1); W = size_U(2); C = size_U(3); N = size_U(4);
% rng('default');U{1} = randn(H, dim);
% rng('default');U{2} = randn(W, dim);
% rng('default');U{3} = randn(C, dim);
% rng('default');U{4} = ones(N, N);
U{1} = randn(H, dim); U{1} = U{1}/norm(U{1},'fro');
U{2} = randn(W, dim); U{2} = U{2}/norm(U{2},'fro');
U{3} = randn(C, dim); U{3} = U{3}/norm(U{3},'fro');
U{4} = ones(N, N);      


end

function [U_mode_x, U_mode_y, corr] = update_U(X,Y,Ux,Uy,dim,mode)

mode_all = 1:3; proj_mode = setdiff(mode_all,mode);
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

function [A,B,sum_r] = solve_CCA(X_mean, Y_mean, mode, dim)

[Hx,Wx,Cx,N] = size(double(X_mean)); [Hy,Wy,Cy,N] = size(double(Y_mean));


Sxx = 0;
for i = 1:N
    Xi = X_mean(:,:,:,i);
    Xi_mode = double(tenmat(Xi,mode));
    Sxx = Sxx + Xi_mode * Xi_mode';
end

[Fx,~] = size(double(Xi_mode));

Sxx = Sxx/N + 0.001 * eye(Fx); % regularized CCA

Syy = 0;
for i = 1:N
    Yi = Y_mean(:,:,:,i);
    Yi_mode = double(tenmat(Yi,mode));
    Syy = Syy + Yi_mode * Yi_mode';
end

[Fy,~] = size(double(Yi_mode));

Syy = Syy/N + 0.001 * eye(Fy); % regularized CCA

Sxy = 0;
for i = 1:N
    Xi = X_mean(:,:,:,i); Yi = Y_mean(:,:,:,i);
    Xi_mode = double(tenmat(Xi,mode)); Yi_mode = double(tenmat(Yi,mode));
    Sxy = Sxy + Xi_mode * Yi_mode';
end
Sxy = Sxy/N;

r_dim = min(Fx, Fy);
r_dim = min(r_dim, dim);

Sxx_inv = Sxx^(-0.5); Syy_inv = Syy^(-0.5);

T = Sxx_inv*Sxy*Syy_inv;
[U,D,V] = svd(T);
r = diag(D);
A = Sxx_inv*U(:,1:r_dim);
B = Syy_inv*V(:,1:r_dim);

sum_r = sum(r);

end


function [A,B,corr] = solve_CCA_new(X_mean,Y_mean, dim, N)

[Fx, ~] = size(X_mean); [Fy, ~] = size(Y_mean);

r_dim = min(Fx, Fy);
r_dim = min(r_dim, dim);

Sxx = (X_mean * X_mean');
Syy = (Y_mean * Y_mean');
Sxy = (X_mean * Y_mean')/N;

Sxx = Sxx/N + 0.01 * eye(Fx); % regularized CCA

Syy = Syy/N + 0.01 * eye(Fy); % regularized CCA

% [Vx,Dx] = eig((Sxx+Sxx')/2);
% [Vy,Dy] = eig((Syy+Syy')/2);
% Dx = diag(Dx);
% Dy = diag(Dy);
% Sxx_inv = Vx*diag(real(Dx.^(-1/2)))*Vx';
% Syy_inv = Vy*diag(real(Dy.^(-1/2)))*Vy';
Sxx_inv = Sxx^(-0.5); Syy_inv = Syy^(-0.5);

T = Sxx_inv*Sxy*Syy_inv;
[U,D,V] = svd(T);
r=diag(D);
A = Sxx_inv*U(:,1:r_dim);
B = Syy_inv*V(:,1:r_dim);

corr = sum(r);

end
