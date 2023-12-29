function [Ux, Uy, Mx, My] = STCCA_Wang(X,Y, lambda_u, lambda_v, dim)
%STCCA 此处显示有关此函数的摘要
%   此处显示详细说明

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

max_iters = 20; epsilon_corr = 0.01;
Ux = initialize_U(size_X, dim);
Uy = initialize_U(size_Y, dim);
sum_corr_old = 0;

for iter = 1:max_iters
    sum_corr = 0;
    for mode = 1: dim_data - 1
        [Ux_mode, Uy_mode, sum_r] = update_U(X,Y,Ux,Uy,mode,lambda_u,lambda_v,dim);
        Ux{mode} = Ux_mode;
        Uy{mode} = Uy_mode;
        sum_corr = sum_corr + sum_r;
    end
    
    sum_corr_new = sum_corr;
    
    if(abs(sum_corr_new - sum_corr_old)<epsilon_corr)
        disp(['converge at ', num2str(iter), ' th iteration']);
        break;
    else
        sum_corr_old = sum_corr_new;
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

function [Ux_mode, Uy_mode, sum_r] = update_U(X,Y,Ux,Uy,mode,lambda_u,lambda_v,dim)

dim_data = ndims(X);

mode_all = 1:dim_data-1; proj_mode = setdiff(mode_all,mode);
N = size(double(X),ndims(double(X)));

X_proj = ttm(tensor(X),Ux,proj_mode,'t');
Y_proj = ttm(tensor(Y),Uy,proj_mode,'t');

X_mode = tenmat(X_proj, mode);
Y_mode = tenmat(Y_proj, mode);

X_mode = double(X_mode);
Y_mode = double(Y_mode);

[Ux_mode, Uy_mode, sum_r] = solve_sparse_CCA(X_mode,Y_mode, dim, N, lambda_u, lambda_v);

end

function [U_new, V_new, sum_r] = solve_sparse_CCA(X_mean,Y_mean, dim, N, lambda_u, lambda_v)

[Fx, ~] = size(X_mean); [Fy, ~] = size(Y_mean); esilon = 0.01;

r_dim = min(Fx, Fy);
r_dim = min(r_dim, dim);

Sxx = (X_mean * X_mean');
Syy = (Y_mean * Y_mean');
Sxy = (X_mean * Y_mean')/N;

if(Fx > N)
    Sxx = Sxx + 0.001 * eye(Fx); % regularized CCA
end
Sxx = Sxx/N;

if(Fy > N)
    Syy = Syy + 0.001 * eye(Fy); % regularized CCA
end
Syy = Syy/N;

[Vx,Dx] = eig((Sxx+Sxx')/2);
[Vy,Dy] = eig((Syy+Syy')/2);
Dx = diag(Dx);
Dy = diag(Dy);
Sxx_inv = Vx*diag(real(Dx.^(-1/2)))*Vx';
Syy_inv = Vy*diag(real(Dy.^(-1/2)))*Vy';

C = Sxx_inv*Sxy*Syy_inv;
[U,D,V] = svd(C);
r=diag(D);
U = U(:,1:r_dim);
V = V(:,1:r_dim);
r = r(1:r_dim);
sum_r = sum(r);

U_new = zeros(size(U)); V_new = zeros(size(V));

for j = 1:r_dim
    u_old = U(:,j); v_old = r(j) * V(:,j);
    for iter = 1:100
        C_u = C'*u_old;
        v_new = sign(C_u) .* max(0, abs(C_u) - lambda_v/2);
        if(norm(v_new) == 0)
            v_new = v_new;
        else
            v_new = v_new/norm(v_new);
        end
        
        C_v = C * v_new;
        u_new = sign(C_v) .* max(0, abs(C_v) - lambda_u/2);
        if(norm(u_new) == 0)
            u_new = u_new;
        else
            u_new = u_new/norm(u_new);
        end
        
        % Check convergance
        if(norm(u_new - u_old) < esilon && norm(v_new - v_old) <esilon)
            break;
        else
            u_old = u_new; v_old = v_new;
        end
    end
    C = C - (u_new' * C * v_new) * (u_new * v_new');
    U_new(:,j) = u_new; V_new(:,j) = v_new;
end

end



