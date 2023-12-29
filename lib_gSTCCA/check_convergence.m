function [convergence_by_Jmax, convergence_flag_by_lambda] = check_convergence(Xten, Xt, Y, alpha, epsilon)
%% Initialize
convergence_flag_by_lambda = 0;
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
    convergence_by_Jmax = 0;
else
    fprintf('J_max is zeros! Terminate now.\n')
    convergence_by_Jmax = 1;
end

lambda(t) = J_max/M - alpha * epsilon;
if(lambda(t) < 0)
    convergence_flag_by_lambda = 1;
end

end

