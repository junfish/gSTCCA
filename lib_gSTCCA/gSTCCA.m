function [U_cell, V_cell, U_new, V_new, running_record] = gSTCCA(X, Y, max_iter, Rx, Ry, epsilon, tolerance, normalize, initialize, t_selected)

N_samples = size(X,ndims(X));
dim_X = ndims(X) - 1;
dim_Y = ndims(Y) - 1;

Mx = mean(X,ndims(X));
My = mean(Y,ndims(Y));

Xc = X - Mx;
Yc = Y - My;

% Use randn for initialization
if(strcmp(initialize,'rand') == 1)
    [init_U] = initialize_U(Xc,Rx);
    [init_V] = initialize_U(Yc,Ry);
end

if(strcmp(initialize,'rand_new') == 1)
    [init_U] = initialize_U_new(Xc,Rx);
    [init_V] = initialize_U_new(Yc,Ry);
end

% Use MPCA for initialization (Faster)
if(strcmp(initialize,'MPCA') == 1)
    [init_U] = MPCA_my(Xc, Rx);
    [init_V] = MPCA_my(Yc, Ry);
end


if(strcmp(initialize,'CP') == 1)
    [init_U] = CP_init(Mx, Rx);
    [init_V] = CP_init(My, Ry);
end


if(strcmp(initialize,'CP') == 0)
    X_proj = cal_proj_init(Xc,init_U,Rx);
    Y_proj = cal_proj_init(Yc,init_V,Ry);
else
    X_proj = cal_proj_init_with_CP(Xc,init_U);
    Y_proj = cal_proj_init_with_CP(Yc,init_V);
end


vec_X = double(tenmat(Xc, ndims(Xc)));
vec_Y = double(tenmat(Yc, ndims(Yc)));

convergence_X_flag_by_lambda = 0;
convergence_Y_flag_by_lambda = 0;
error_U = 200; error_V = 200;
running_record{1,3} = error_U;
running_record{1,4} = error_V;

error_list_U(1) = error_U;
error_list_V(1) = error_V;

early_stop_U = 0; early_stop_V = 0;

for i = 1:max_iter
        
    % update U
    if(convergence_X_flag_by_lambda == 0 && early_stop_U == 0)
        [U_lambda_all,~,U_t,U_cell, convergence_X_flag_by_lambda] = gSTCCA_simple(Xc, vec_X, Y_proj, Rx, epsilon, t_selected);
        U_new = cal_U_sum(U_t);
        X_proj = cal_proj(Xc, U_new);
    end
    
    if(convergence_X_flag_by_lambda == 1)
        disp(['converge at the ',num2str(i),' -th iteration'])
        break;
    end
    
    % update V
    if(convergence_Y_flag_by_lambda == 0 && early_stop_V == 0)
        [V_lambda_all,~,V_t,V_cell, convergence_Y_flag_by_lambda] = gSTCCA_simple(Yc, vec_Y, X_proj, Ry, epsilon, t_selected);
        V_new = cal_U_sum(V_t);
        Y_proj = cal_proj(Yc, V_new);
    end
    
    if(convergence_Y_flag_by_lambda == 1)
        disp(['converge at the ',num2str(i),' -th iteration'])
        break
    end
    
    if(early_stop_U == 1 && early_stop_V == 1)
        disp(['converge at the ',num2str(i),' -th iteration'])
        break;
    end
    
    if(i == 1)
        U_old = U_new;
        V_old = V_new;
    end
    
    running_record{i,1} = U_lambda_all;
    running_record{i,2} = V_lambda_all;
    
    
    if(i > 1)
        error_U = norm(U_new(:) - U_old(:),'fro')/norm(U_new(:));
        error_V = norm(V_new(:) - V_old(:),'fro')/norm(V_new(:));
        
        running_record{i,3} = error_U;
        running_record{i,4} = error_V;
        
        error_list_U(i) = error_U;
        error_list_V(i) = error_V;       
        
        if(early_stop_U == 0)
            early_stop_U = check_early_stop(error_list_U);
        end
        
        if(early_stop_V == 0)
            early_stop_V = check_early_stop(error_list_V);
        end
        
        if(error_U < tolerance && error_V < tolerance)
            disp(['model converge at ',num2str(i),'-th iteration']);
            break;
        else
            U_old = U_new; V_old = V_new;
        end
    end
    
end

if(normalize == 1)
    U_cell_new = cell(dim_X,1);
    V_cell_new = cell(dim_Y,1);
    
    for i = 1:dim_X
        Ui = U_cell{i};
        Rx = size(Ui,2);
        for r = 1:Rx
            Xi_r = cal_proj_i_r(Xc, U_cell, i, r);
            Ui_r = U_cell{i}(:,r);
            U_cell_new{i}(:,r) = normalize_proj_vec(Ui_r, Xi_r);
        end
        
        Vi = V_cell{i};
        Ry = size(Vi,2);
        for r = 1:Ry
            Yi_r = cal_proj_i_r(Yc, V_cell, i, r);
            Vi_r = V_cell{i}(:,r);
            V_cell_new{i}(:,r) = normalize_proj_vec(Vi_r, Yi_r);
        end
    end
    
    U_cell = U_cell_new; V_cell = V_cell_new;
end


end

function U_normalize = normalize_proj_vec(Ui_r, Xi_r)

var_UX = var(Ui_r' * Xi_r);
U_normalize = var_UX^(-0.5) * Ui_r;

end

function Xi_r = cal_proj_i_r(X,U,mode,r)

dim_X = ndims(X) - 1;
U_r = cell(dim_X,1);
for i = 1:dim_X
    Ui_r = U{i}(:,r);
    U_r{i} = Ui_r;
end

mode_all = 1:dim_X;
mode_proj = setdiff(mode_all, mode);

Xi_r = ttm(tensor(X), U_r(mode_proj), mode_proj,'t');
Xi_r = squeeze(double(Xi_r));

end

function U = initialize_U(X,R)
N = ndims(X)-1;
size_X = size(X);
U = cell(N,1);

for i = 1:N
    rng('default');Ui = randn(size_X(i), R);
    U{i} = Ui;
    %   U{i} = Ui/norm(Ui,1);
end

end


function U = initialize_U_new(X,R)

N = ndims(X)-1;
U = cell(N,1);
size_X = size(X);
for n=1:N
    U{n} = rand(size_X(n),R);
    for j=1:R
        U{n}(:,j) = U{n}(:,j) / norm(U{n}(:,j));
    end
end

end

function X_proj = cal_proj_init(X,U,R)
N = ndims(X);
mode = 1:N-1;
U_r = cell(N-1,1);

sum_proj = 0;
for r = 1:R
    
    for j = 1:N-1
        U_r{j} = U{j}(:,r);
    end
    
    proj_i = double(ttm(tensor(X),U_r,mode,'t'));
    sum_proj = sum_proj + proj_i;
end

X_proj = squeeze(sum_proj);

end

function X_proj = cal_proj_init_with_CP(X,U)
N_samples = size(X, ndims(X));
X_proj = zeros(N_samples,1);
ndims_X = ndims(X);
vec_U = U(:);

for i = 1:N_samples
    if(ndims_X == 3)
        Xi = X(:,:,i);
        vec_Xi = Xi(:);
    else
        Xi = X(:,:,:,i);
        vec_Xi = Xi(:);
    end
    X_proj(i) = vec_Xi' * vec_U;
end

end

function U_sum = cal_U_sum(U)
R = length(U);
U_sum = 0;

for r = 1:R
    Ur = U{r};
    if(~isempty(Ur))
        U_sum = U_sum + Ur;
    end
end

end

function X_proj = cal_proj(X, U)
N = size(X,ndims(X));
X_proj = zeros(N,1);

for i = 1:N
    
    if(ndims(X) == 3)
        Xi = X(:,:,i);
    elseif(ndims(X) == 4)
        Xi = X(:,:,:,i);
    end
    
    Xi_U = Xi.*U;
    X_proj_i = sum(Xi_U(:));
    X_proj(i,1) = X_proj_i;
end

end

function [U] = MPCA_my(X, r)
%MPCA 此处显示有关此函数的摘要
%   此处显示详细说明
dim_X = ndims(X);X = tensor(X);
for i = 1:dim_X-1
    Xi = double(tenmat(X,i));
    Ui = cal_U(Xi,r);
    U{i} = Ui;
end

end

function U = cal_U(X,rank)
[U,S,~] = svd(X*X');
U = U(:,1:rank);

end

function U = CP_init(X, rank)

[P,~,~] = cp_als(tensor(X),rank);
lambda = P.lambda; U_cell = P.U;
[U] = cal_CP_U(lambda,U_cell);

end

function early_stop = check_early_stop(error_list)

l_error = length(error_list);
error_new = error_list(l_error);
error_old = error_list(l_error-1);

if(error_new > error_old)
    early_stop = 1;
else
    early_stop = 0;
end

end


