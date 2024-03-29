function  [w_all, w_vector, W_tensor, w_final, sigma_final, res, lambda] = Train_gSTCCA(Xten,Xt,Y,epsilon,max_iter,xi,alpha,t_selected)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% optiLambda or absW
%% Set defalt parameters
if nargin < 3
    fprintf('Not enough input! Terminate now.\n')
    return;
end

%% Initialize
N = ndims(Xten) - 1;         % Number of modes
M = size(Xten,N+1);
sz = size(Xten);
for i=1:N+1
    DIM{1,i} = 1:sz(i);
end
dim = sz(1:N);
mode = cumsum(ones(1,N));

%% Initialize solution by {w} = (s1, 1, ..., 1)
t = 1;
XY = Xt' * Y;
diag_XX = sum(Xt.^2, 1);
[J_max, ind_max] = max(2*abs(XY)-diag_XX'*epsilon);
if J_max~=0
    s = sign(XY(ind_max)) * epsilon;
else
    fprintf('J_max is zeros! Terminate now.\n')
    return;
end

[out{1:N}] = ind2sub(dim,ind_max);
index = cell2mat(out);

for i=1:N
    I{1,i} = index(i);
    w{t,i} = zeros(1,dim(i));
    w{t,i}(I{1,i}) = 1;
end

sigma(t) = epsilon;
w{t,1} = sign(s) * w{1,1};
lambda(t) = J_max/M - alpha * epsilon;

%% Calculate Z -- Z^{(-n)}
Z = cell(1,N);
for i=1:N
    inds_n = DIM;
    mode_n = setdiff(mode,i);
    inds_n(mode_n) = I(mode_n);
    if i==1
        Z{i} = squeeze(Xten(inds_n{:}));
    else
        Z{i} = squeeze(Xten(inds_n{:}))* sign(s);
    end
end

%% Trace out solution path
while t<max_iter
    [m_hat, i_hat, s_hat, delta_Gamma, Ze, Zd,e] = backward_index(Z, Y, sigma(t), w(t,:), dim, I, alpha, epsilon, mode);
    if m_hat~= 0 && -delta_Gamma <= lambda(t) * epsilon - xi
%         fprintf('The %g -th backward optimamum is: m_hat=%g, i_hat = %g, s_hat=%g \n', t, m_hat, i_hat,s_hat);
        w_hat = w{t,m_hat} * sigma(t);
        [sigma_update, w_update, I] = backward_update(w_hat, m_hat, i_hat, s_hat, I);
        sigma(t+1) = sigma_update;
        w(t+1,:) = w(t,:);
        w{t+1,m_hat} = w_update;
        lambda(t+1) = lambda(t);     
    else
        [m_hat,i_hat,s_hat, delta_J] = forward_index(Ze, Zd, dim, epsilon, M);
%         fprintf('The %g -th forward is: m_hat=%g, i_hat = %g, s_hat=%g \n', t, m_hat, i_hat, s_hat);
        if m_hat~=0
            w_hat = w{t,m_hat} * sigma(t);
            [sigma_update, w_update, I] = forward_update_w(w_hat, m_hat, i_hat, s_hat, I);
            if sigma_update==0
                fprintf('Sigma equals to zero! Terminate now.\n')
                break;
            else
                sigma(t+1) = sigma_update;
                w(t+1,:) = w(t,:);
                w{t+1,m_hat} = w_update;
                lambda(t+1) =  forward_update_lambda(delta_J, lambda(t), epsilon, xi);
            end
            
            if lambda(t+1) < 0
                fprintf('Lambda =%g! Terminate now.\n',lambda(t+1));
                lambda(t+1) = [];
                w(t+1,:) = [];
                sigma(t+1) = [];
                break;
            end
            
        else
            fprintf('No any further index is selected! Terminate now.\n')
            break;
        end
    end
    
    %% Update Z_n
    inds_n = DIM;
    inds_n{m_hat} = i_hat;
    Xten_m = Xten(inds_n{:});
    if N==2
        j = setdiff(mode,m_hat);
        INC = squeeze(Xten_m) * s_hat;
        Z{j} = sigma(t) * Z{j} + INC;
        Z{j} = Z{j}/sigma(t+1);
    else
        for i = 1:N
            if i~=m_hat
                mode_n = setdiff(mode,[i,m_hat]);
                tmp = tmprod(Xten_m,w(t+1,mode_n),mode_n);
                INC = squeeze(tmp) * s_hat;
                Z{i} = sigma(t) * Z{i} + INC;
                Z{i} = Z{i}/sigma(t+1);
            end
        end
    end
    t = t+1;
end

t_selected = min(t_selected,t);

w_final = w(t_selected,:);
sigma_final = sigma(t_selected);
W_tensor = cellfun(@transpose,w_final,'un',0);
W_tensor = cpdgen(W_tensor) * sigma_final;
w_vector = W_tensor(:);
a = ttm(tensor(Xten), w_final,1:N,'nt');
a = double(a); a = squeeze(a); a = a*sigma_final;

res = Y - a;
w_all = w;


end


function [w_hat_final] = cal_final_w(X, w_all, w_tilta, n)

dim_x = ndims(X) - 1;
N = size(X,ndims(X));
mode = 1:dim_x;
mode_res = setdiff(mode, n);
w_res = w_all{:,mode_res};
X_minus_n = ttm(tensor(X), w_res, mode_res, 'nt');
X_minus_n = squeeze(double(X_minus_n));
var_x_n = var(w_tilta*X_minus_n);
w_hat_final = var_x_n^(-0.5) * w_tilta;


end

function proj = cal_proj(X, W)
N = size(X,ndims(X));
w = W(:);

for i = 1:N
    if(ndims(X) == 4)
        Xi = X(:,:,:,i);
    else
        Xi = X(:,:,i);
    end
    xi = Xi.*W;
    sum_i = sum(xi(:));
    proj(i) = sum_i;
end

end
