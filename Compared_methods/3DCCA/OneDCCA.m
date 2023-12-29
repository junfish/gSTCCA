function [A,B,r,U,V] = OneDCCA(X,Y,mode,solution)
%CCA_MY 此处显示有关此函数的摘要
%   此处显示详细说明

if(strcmp(solution,'EVD') == 1)
     [A,B, r, U, V] = solve_EVD(X,Y,mode);
end

if(strcmp(solution,'SVD') == 1)
     [A,B, r, U, V] = solve_SVD(X,Y,mode);
end

end

function X_project = reduce_dimension(X)
r = 16;
[~, N] = size(X);

if(r>N)
    r = 6;
end

[U,~,~] = svds(X, r);
X_project = U'*X;

end

function U_sort = rerank(U, S)

[N,~] = size(S);
idx = N:-1:1;
U_sort = U(:,idx);

end

function [A,B, r, U, V] = solve_EVD(X,Y,mode)

X_mean = X - mean(X,2);
Y_mean = Y - mean(Y,2);

[Fx, N] = size(X); [Fy, N] = size(Y);

if(Fx > N)
    if(strcmp(mode, 'PCA') == 1)
        X_mean = reduce_dimension(X_mean);
        [Fx,~] = size(X_mean);
    end
end

if(Fy > N)
    if(strcmp(mode, 'PCA') == 1)
        Y_mean = reduce_dimension(Y_mean);
        [Fy,~] = size(Y_mean);
    end
end

r_dim = min(Fx, Fy);

Sxx = (X_mean * X_mean')/N;
Syy = (Y_mean * Y_mean')/N;
Sxy = (X_mean * Y_mean')/N;
Syx = Sxy';

if(Fx > N)
    Sxx = Sxx + 0.1 * eye(Fx); % regularized CCA
end

if(Fy > N)
    Syy = Syy + 0.1 * eye(Fy); % regularized CCA
end

inv_Sxx = inv(Sxx); inv_Syy = inv(Syy);
M_A = inv_Sxx * Sxy * inv_Syy * Syx;
M_B = inv_Syy * Syx * inv_Sxx * Sxy;

[A,S_A] = eigs(M_A, r_dim);
[B,S_B] = eigs(M_B, r_dim);

error = diag(S_A) - diag(S_B);
if(norm(error) > 0.001)
    disp('error computing')
end

diag_S = diag(S_A);
r = sqrt(diag_S);

U = A'*X_mean; V = B'*Y_mean;

end

function [A,B, r, U, V] = solve_SVD(X,Y, mode)

X_mean = X - mean(X,2);
Y_mean = Y - mean(Y,2);

[Fx, N] = size(X); [Fy, N] = size(Y);

if(Fx > N)
    if(strcmp(mode, 'PCA') == 1)
        X_mean = reduce_dimension(X_mean);
        [Fx,~] = size(X_mean);
    end
end

if(Fy > N)
    if(strcmp(mode, 'PCA') == 1)
        Y_mean = reduce_dimension(Y_mean);
        [Fy,~] = size(Y_mean);
    end
end

r_dim = min(Fx, Fy);

Sxx = X_mean * X_mean';
Syy = Y_mean * Y_mean';
Sxy = X_mean * Y_mean';

if(Fx > N)
    Sxx = Sxx + 0.001 * eye(Fx); % regularized CCA
end

if(Fy > N)
    Syy = Syy + 0.001 * eye(Fy); % regularized CCA
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

U = A'*X_mean; V = B'*Y_mean;

end

