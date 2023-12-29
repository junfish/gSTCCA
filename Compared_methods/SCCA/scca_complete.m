function [U,V,mx,my] = spls_complete(X,Y,cu,cv,R,max_iter)
%SPLS_COMPLETE 此处显示有关此函数的摘要
%   此处显示详细说明
mx = mean(X,1); my = mean(Y,1);

X = X - mean(X,1);
Y = Y - mean(Y,1);
X_res = X; Y_res = Y;

for r = 1:R
    [u_r,v_r] = spls(X_res, Y_res, cu, cv, 0.00001, max_iter);
    [X_res, Y_res] = proj_def(X_res, Y_res, u_r, v_r);
    
    U(:,r) = u_r; V(:,r) = v_r;
end

end

