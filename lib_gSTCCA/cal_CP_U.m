function [U_tensor] = cal_CP_U(lambda,U_cell)
%CAL_CP_U 此处显示有关此函数的摘要
%   此处显示详细说明

R = length(lambda); 

U_tensor = 0;
for r = 1:R
    lambda_r = lambda(r);
    U_r = cal_U_R(lambda_r,U_cell,r);
    U_tensor = U_tensor + U_r;
end

end

function U_r = cal_U_R(lambda_r,U_cell,r)

N = length(U_cell);
if(N == 2)
    a = U_cell{1}(:,r);
    b = U_cell{2}(:,r);
    U_r = a * b';
end

if(N == 3)
    a = U_cell{1}(:,r);
    b = U_cell{2}(:,r);
    c = U_cell{3}(:,r);
    
    l_a = length(a); l_b = length(b); l_c = length(c);
    U_r = zeros(l_a, l_b, l_c);
    for i = 1:l_a
        for j = 1:l_b
            for k = 1:l_c
                U_r(i,j,k) = a(i) * b(j) * c(k);
            end
        end
    end
end

U_r = U_r * lambda_r;


end

