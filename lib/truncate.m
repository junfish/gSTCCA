function V_r = truncate(X, r)
[N,F] = size(X); dim_svds = min(N,F);
[~, Sx, Vx] = svds(X,dim_svds);
diag_S = diag(Sx);
sum_diag_S = 0;
sum_all_S = sum(diag_S);

for i = 1:length(diag_S)
    Si = diag_S(i);
    sum_diag_S = sum_diag_S + Si;
    if(sum_diag_S/sum_all_S>=r)
        break;
    end
end

V_r = Vx(:,1:i);

end