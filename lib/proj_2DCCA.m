function X_proj = proj_2DCCA(X,L,R)
if(ndims(X) == 2)
    X_proj = L'*X*R;
else
    N = size(X,ndims(X));
    for i = 1:N
        Xi = X(:,:,i);
        X_proj_i = L'*Xi*R;
        X_proj(:,:,i) = X_proj_i;
    end
end

end