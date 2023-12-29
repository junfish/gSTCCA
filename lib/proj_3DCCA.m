function X_proj = proj_3DCCA(X,U)
X_proj = ttm(tensor(X),U,[1,2,3],'t'); X_proj = double(X_proj);
end