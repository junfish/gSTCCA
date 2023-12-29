function canvar = aaa(X, Ux)
[~,r] = size(Ux{1});dim_x = ndims(X);
mode_proj = 1:dim_x-1;

canvar = 0;
for i = 1:r
    Ui = fetch_U(Ux, i);
    canvar_i = ttm(tensor(X), Ui, mode_proj,'t');
    canvar_i = double(squeeze(canvar_i));
    canvar = canvar + canvar_i;
end

end

function Ur = fetch_U(U,r)

dim_x = length(U); Ur = cell(dim_x,1);
for i = 1:dim_x
    Ui = U{i};
    Ur_i = Ui(:,r);
    Ur{i} = Ur_i;
end

% Ur{dim_x} = U{dim_x};

end