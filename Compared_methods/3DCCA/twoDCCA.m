function [Lx,Ly,Rx,Ry, Mx, My] = twoDCCA(X,Y,d1,d2,solution)
%2DCCA 此处显示有关此函数的摘要
%   X and Y are 3D tensors HxWxN consisting of N samples.
%   Lx: Hx x d1, Ly: Hy x d1; Rx: Wx x d2, Ly: Wy x d2;

% Initialize
[Hx, Wx, Nx] = size(X); [Hy, Wy, Ny] = size(Y);
if(Nx ~= Ny)
    disp('wrong number of samples')
    return
end

% Remove mean
Mx=mean(X,3);
My=mean(Y,3);
X=bsxfun(@minus,X,Mx);
Y=bsxfun(@minus,Y,My);

max_iters = 20;
l_dim = min(d1,min(Hx,Hy));
r_dim = min(d2,min(Wx,Wy));
% rng('default'); Rx=randn(Wx,r_dim);Rx = Rx/norm(Rx);
% rng('default'); Ry=randn(Wy,r_dim);Ry = Ry/norm(Ry);
% rng('default'); Lx=randn(Hx,l_dim);Lx = Lx/norm(Lx);
% rng('default'); Ly=randn(Hy,l_dim);Ly = Ly/norm(Ly);

Rx=randn(Wx,r_dim);Rx = Rx/norm(Rx);
Ry=randn(Wy,r_dim);Ry = Ry/norm(Ry);
Lx=randn(Hx,l_dim);Lx = Lx/norm(Lx);
Ly=randn(Hy,l_dim);Ly = Ly/norm(Ly);

corr_old = 0;
if(strcmp(solution,'SVD') == 1)
    for i = 1:max_iters
        % Cal Lx and Ly
        [Lx, Ly,corr_L] = update_L(X,Y,Rx,Ry,l_dim);
        % Cal Rx and Ry
        [Rx, Ry,corr_R] = update_R(X,Y,Lx,Ly,r_dim); 
%         corr_new = corr_L + corr_R;
%         if(abs(corr_new - corr_old) < 0.005)
%             break;
%         else
%             corr_old = corr_new;
%         end
    end
end

if(strcmp(solution,'EVD') == 1)
    for i = 1:max_iters
        % Cal Lx and Ly using EVD
        [Lx, Ly,corr] = update_L_EVD(X,Y,Rx,Ry,l_dim);
        % Cal Rx and Ry using EVD
        [Rx, Ry,corr] = update_R_EVD(X,Y,Lx,Ly,r_dim); 
    end
end

end


function [Lx,Ly,corr]=update_L(X,Y,Rx,Ry,dim)

[mx,~,N] =size(X);
[my,~,~] =size(Y);

% Compute the auto covariance matrices
s=zeros(mx,my);
for i=1:N
    s=s+X(:,:,i)*(Rx*Ry')*Y(:,:,i)';
end
Sxy = (1.0/N)*s;

% Obtain Sxx
s=zeros(mx,mx);
for i=1:N
    s=s+X(:,:,i)*(Rx*Rx')*X(:,:,i)';
end

Sxx = (1.0/N)*s+(1e-3)*eye(mx);

% Obtain Syy
s=zeros(my,my);
for i=1:N
    s=s+Y(:,:,i)*(Ry*Ry')*Y(:,:,i)';
end
Syy = (1.0/N)*s+(1e-3)*eye(my);

Sxx_inv = Sxx^(-0.5);Syy_inv = Syy^(-0.5);
Sxx_inv = real(Sxx_inv); Syy_inv = real(Syy_inv);

% Singular value decomposition
T = Sxx_inv*Sxy*Syy_inv;
[U,D,V] = svd(T);
D=diag(D);
Lx = Sxx_inv*U(:,1:dim);
Ly = Syy_inv*V(:,1:dim);
D = D(1:dim);
corr=sum(D);

end

function [Rx,Ry,corr]=update_R(X,Y,Lx,Ly,dim)
[~,nx,N] =size(X);
[~,ny,~] =size(Y);

% Compute the auto covariance matrices
s=zeros(nx,ny);
for i=1:N
    s=s+X(:,:,i)'*(Lx*Ly')*Y(:,:,i);
end
Kxy = (1.0/N)*s;

% Obtain Kxx
s=zeros(nx,nx);
for i=1:N
    s=s+X(:,:,i)'*(Lx*Lx')*X(:,:,i);
end
Kxx = (1.0/N)*s+(1e-3)*eye(nx);

% Obtain Kyy
s=zeros(ny,ny);
for i=1:N
    s=s+Y(:,:,i)'*(Ly*Ly')*Y(:,:,i);
end
Kyy = (1.0/N)*s+(1e-3)*eye(ny);

Kxx_inv = Kxx^(-0.5);Kyy_inv = Kyy^(-0.5);
Kxx_inv = real(Kxx_inv); Kyy_inv = real(Kyy_inv);

% Singular value decomposition
T = Kxx_inv*Kxy*Kyy_inv;
[U,D,V] = svd(T);
D=diag(D);
Rx = Kxx_inv*U(:,1:dim);
Ry = Kyy_inv*V(:,1:dim);
D = D(1:dim);
corr=sum(D);

end

function [Lx, Ly, corr] = update_L_EVD(X,Y,Rx,Ry,l_dim)

[Hx, Wx, Nx] = size(X); [Hy, Wy, Ny] = size(Y);

if(Nx ~= Ny)
    disp('wrong number of samples')
    return
end

N = Nx;

Sr_xx = cal_Sr(X, Rx, Rx, X);
Sr_xy= cal_Sr(X, Rx, Ry, Y);
Sr_yx = Sr_xy';
Sr_yy= cal_Sr(Y, Ry, Ry, Y);

if(Hx > N)
    Sr_xx = Sr_xx + 0.1*ones(Hx);
    disp('using regularized CCA')
end

if(Hy > N)
    Sr_yy = Sr_yy + 0.1*ones(Hy);
    disp('using regularized CCA')
end

inv_Sr_xx = Sr_xx^(-1); inv_Sr_yy = Sr_yy^(-1); 
M_A = inv_Sr_xx * Sr_xy * inv_Sr_yy * Sr_yx;
M_B = inv_Sr_yy * Sr_yx * inv_Sr_xx * Sr_xy;

[Lx,Dx] = eigs(M_A, l_dim);
[Ly,Dy] = eigs(M_B, l_dim);

error = diag(Dx) - diag(Dy);
if(norm(error) > 0.001)
    disp('error computing');
end

D = sqrt(diag(Dx));
corr = sum(D);

end

function [Rx, Ry, corr] = update_R_EVD(X,Y,Lx,Ly,r_dim)

[Hx, Wx, Nx] = size(X); [Hy, Wy, Ny] = size(Y);

if(Nx ~= Ny)
    disp('wrong number of samples')
    return
end

N = Nx;

Sl_xx = cal_Sl(X, Lx, Lx, X);
Sl_xy = cal_Sl(X, Lx, Ly, Y);
Sl_yx = Sl_xy';
Sl_yy= cal_Sl(Y, Ly, Ly, Y);

if(Wx > N)
    Sl_xx = Sl_xx + 0.1*ones(Wx);
    disp('using regularized CCA')
end

if(Wy > N)
    Sl_yy = Sl_yy + 0.1*ones(Wy);
    disp('using regularized CCA')
end

inv_Sl_xx = Sl_xx^(-1); inv_Sl_yy =  Sl_yy^(-1); 
M_A = inv_Sl_xx * Sl_xy * inv_Sl_yy * Sl_yx;
M_B = inv_Sl_yy * Sl_yx * inv_Sl_xx * Sl_xy;

[Rx,Dx] = eigs(M_A, r_dim);
[Ry,Dy] = eigs(M_B, r_dim);

error = diag(Dx) - diag(Dy);
if(norm(error) > 0.001)
    disp('error computing');
end

D = sqrt(diag(Dx));
corr = sum(D);

end

function S = cal_Sr(X, U1, U2, Y)

[~,~,N] = size(X);
S = 0;
for i = 1:N
    Xi = X(:,:,i); Yi = Y(:,:,i);
    Si = Xi * U1 * U2' * Yi';
    S = S + Si;
end

S = S/N;

end

function S = cal_Sl(X, U1, U2, Y)

[~,~,N] = size(X);
S = 0;
for i = 1:N
    Xi = X(:,:,i); Yi = Y(:,:,i);
    Si = Xi' * U1 * U2' * Yi;
    S = S + Si;
end

S = S/N;

end

