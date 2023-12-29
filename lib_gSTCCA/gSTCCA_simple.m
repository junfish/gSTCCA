function [w_lambda_all, optW, W_tensor,W_final_all, convergence_flag_by_lambda] = gSTCCA_simple(Xten,vec_X,Y,R,epsilon, t_selected)
%UNTITLED4 Summary of this function goes here
%% Outputs
%   optW
%   resSD         The standard deviation of residual

% addpath('tensor_toolbox/');

%% Parameter Setting
% rng('default');               % For reproducibility
rng(1,'twister')
alpha = 1;                      % L2-norm regularized parameter
MaxIter = 200;
dim = size(Xten);
xi = epsilon*0.5;
N_dim = ndims(Xten) - 1;
%% Solve with deflation
lastSD = 10000;
threshold = 1e-4;
optW = zeros(1,prod(dim(1:end-1)));
resSD = sqrt(mse(Y));
W_tensor = cell(R,1);
W_final_all = cell(N_dim,1);

for r = 1:R
    
    [convergence_by_Jmax, convergence_flag_by_lambda] = check_convergence(Xten, vec_X, Y, alpha, epsilon);
    if(convergence_by_Jmax == 1 || convergence_flag_by_lambda == 1)
        break;
    end
    
    [w_all, W, W_tensor_r, w_final, sigma_final, res, lambda_all] = Train_gSTCCA(Xten,vec_X,Y,epsilon,MaxIter,xi,alpha,t_selected);
    
    
    if sum(W) == 0
        fprintf('sum_W equal to zero \n');
%         pause(3)
        break;
    else
        Y = res;
        resSD(r) = sqrt(mse(Y));            % The Standard Deviation of Residual (divide by M, not M-1)
    end
    
    if abs(resSD(r)-lastSD)^2 < threshold
        fprintf('The residual standard deviation starts increase \n');
%         pause(3)
        break;
    else
        optW(r,:) = W;
        W_tensor{r} = W_tensor_r;
        lastSD = resSD(r);
    end
    
    for i = 1:N_dim
        W_final_all{i}(:,r) = w_final{i}';
    end
    
    w_lambda_all{r,1} = w_all;
    w_lambda_all{r,2} = lambda_all;
    w_lambda_all{r,3} = res;
    
    sigma_final_all(r) = sigma_final;
   
end

end

