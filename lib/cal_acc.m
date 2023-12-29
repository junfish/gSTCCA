function acc = cal_acc(estimate, gt)
N = length(gt);
sum_acc = 0;
for i = 1:N
    if(estimate(i) == gt(i))
        sum_acc = sum_acc + 1;
    end
end
acc = sum_acc / N;

end