function [Lx,Ly,Rx,Ry,corr,Mx,My] = twoDCCA_youlin(X, Y, dim, max_iters)

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

if(dim == 1)
    method = 'one_step';
else
    method = 'deflation';
end

l_dim = min(dim,min(Hx,Hy));
r_dim = min(dim,min(Wx,Wy));
% rng('default'); Rx=randn(Wx,r_dim);Rx = Rx/norm(Rx(:));
% rng('default'); Ry=randn(Wy,r_dim);Ry = Ry/norm(Ry(:));
% rng('default'); Lx=randn(Hx,l_dim);Lx = Lx/norm(Lx(:));
% rng('default'); Ly=randn(Hy,l_dim);Ly = Ly/norm(Ly(:));

Rx=randn(Wx,r_dim);Rx = Rx/norm(Rx(:));
Ry=randn(Wy,r_dim);Ry = Ry/norm(Ry(:));
Lx=randn(Hx,l_dim);Lx = Lx/norm(Lx(:));
Ly=randn(Hy,l_dim);Ly = Ly/norm(Ly(:));

corr_old = -1;
for iter = 1:max_iters
    if(dim == 1)
        [Lx,Ly,Rx,Ry] = update_LR_onestep(X,Y,Lx,Ly,Rx,Ry);
        corr = cal_corr(X,Y,Lx,Ly,Rx,Ry);
        corr_new = corr;
        if(corr_new - corr_old < 0.001)
            disp(['converge at iter: ', num2str(iter)])
            break;
        else
            corr_old = corr_new;
        end
    else
        [Lx,Ly,Rx,Ry] = update_LR_deflation(X,Y,Lx,Ly,Rx,Ry,dim);
    end
end

if(dim == 1)
    corr = cal_corr(X,Y,Lx,Ly,Rx,Ry);
else
    corr = 0;
end

end

function [Lx,Ly,Rx,Ry] = update_LR_onestep(X,Y,Lx,Ly,Rx,Ry)

% Cal Lx and Ly
[Lx, Ly] = update_L_onestep(X,Y,Lx,Ly,Rx,Ry);

% Cal Rx and Ry
[Rx, Ry] = update_R_onestep(X,Y,Lx,Ly,Rx,Ry);

end

function [Lx,Ly]=update_L_onestep(X,Y,Lx,Ly,Rx,Ry)

[mx,~,N] =size(X);
[my,~,~] =size(Y);

% Compute the auto covariance matrices
s=zeros(mx,my);
for i=1:N
    s=s+(X(:,:,i)*Rx)*(Ry'*Y(:,:,i)');
end
Sxy = (1.0/N)*s;

% Obtain Sxx
s=zeros(mx,mx);
for i=1:N
    s=s+(X(:,:,i)*Rx)*(Rx'*X(:,:,i)');
end
Sxx = (1.0/N)*s+(1e+0)*eye(mx);

% Obtain Sxx
s=zeros(my,my);
for i=1:N
    s=s+(Y(:,:,i)*Ry)*(Ry'*Y(:,:,i)');
end
Syy = (1.0/N)*s+(1e-2)*eye(my);

% Update Lx and Ly
Lx = Sxx\(Sxy * Ly);
Lx = Lx/norm(Lx);

Ly = Syy\(Sxy' * Lx);
Ly = Ly/norm(Ly);

end

function [Rx, Ry] = update_R_onestep(X,Y,Lx,Ly,Rx,Ry)
[~,nx,N] =size(X);
[~,ny,~] =size(Y);

% Compute the auto covariance matrices
s=zeros(nx,ny);
for i=1:N
    s=s+X(:,:,i)'*(Lx*Ly')*Y(:,:,i);
end
Kxy = (1.0/N)*s;

s=zeros(nx,nx);
for i=1:N
    s=s+X(:,:,i)'*(Lx*Lx')*X(:,:,i);
end
Kxx = (1.0/N)*s+(1e+0)*eye(nx);

s=zeros(ny,ny);
for i=1:N
    s=s+Y(:,:,i)'*(Ly*Ly')*Y(:,:,i);
end
Kyy = (1.0/N)*s+(1e-2)*eye(ny);

Rx = Kxx\(Kxy * Ry);
Rx = Rx/norm(Rx);

Ry =Kyy\(Kxy' * Rx);
Ry = Ry/norm(Ry);


end

function corr = cal_corr(X,Y,Lx,Ly,Rx,Ry)

[~,~,N] = size(X);
vec_x = zeros(N,1);
vec_y = zeros(N,1);

for i = 1:N
    Xi = X(:,:,i);
    vec_x_i = Lx' * Xi * Rx;
    vec_x(i) = vec_x_i;
end

for i = 1:N
    Yi = Y(:,:,i);
    vec_y_i = Ly' * Yi * Ry;
    vec_y(i) = vec_y_i;
end

corr = vec_x'*vec_y/(sqrt(vec_x'*vec_x)*sqrt(vec_y'*vec_y));


end

function [Lx, Ly, Rx, Ry] = update_LR_deflation(X,Y,Lx,Ly,Rx,Ry,rank)

for k = 1:rank
   [X_res, Y_res] = cal_res(X,Y,Lx,Ly,Rx,Ry,k,rank); 
   for t = 1:6
       [Lx_k,Ly_k,Rx_k,Ry_k] = update_LR_onestep(X_res,Y_res,Lx(:,k),Ly(:,k),Rx(:,k),Ry(:,k));
       Lx(:,k) = Lx_k; Ly(:,k) = Ly_k; Rx(:,k) = Rx_k; Ry(:,k) = Ry_k;
   end
end

end

function [X_res, Y_res]  = cal_res(X,Y,Lx,Ly,Rx,Ry,k,rank)
idx_all = 1:rank;
idx_res = setdiff(idx_all, k);
N = size(X,3);
Tk = zeros(N, rank-1); Sk = zeros(N, rank-1);

for i = 1:rank-1
    r = idx_res(i);
    
    Lx_r = Lx(:,r); Rx_r = Rx(:,r);
    Ly_r = Ly(:,r); Ry_r = Ry(:,r);
    
    col_i = ttm(tensor(X),{Lx_r, Rx_r,eye(N)},[1,2],'t'); col_i = double(col_i);
    Tk(:,i) = col_i;
%     Tk(:,i) = cal_col(X,Lx_r, Rx_r);
    
    col_i = ttm(tensor(Y),{Ly_r, Ry_r,eye(N)},[1,2],'t'); col_i = double(col_i);
    Sk(:,i) = col_i;
%     Sk(:,i) = cal_col(Y, Ly_r, Ry_r);
end

T = Tk*Tk'; It = eye(size(T)); T_res = It - T;
X_res = ttm(tensor(X),T_res,3);

S = Sk*Sk'; Is = eye(size(S)); S_res = Is - S;
Y_res = ttm(tensor(Y),S_res,3);

X_res = double(X_res); Y_res = double(Y_res);

% disp('twoDCCA')
% norm(X_res(:)), norm(Y_res(:))


end

function col = cal_col(X, L, R)
N = size(X,3);
col = zeros(N,1);
for i = 1:N
    X_i = X(:,:,i);
    col_i = L'*X_i*R;
    col(i) = col_i;
end

end

