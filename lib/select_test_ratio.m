function idx_test_select = select_test_ratio(label, ratio_test_per_class, rng_factor)
class_all = unique(label);
num_classes = length(class_all);
idx_test_select = [];
for i = 1:num_classes
    class_i = class_all(i);
    idx_class_i = find(label == class_i);
    num_class_i = length(idx_class_i);
    num_test_class_i = floor(ratio_test_per_class * num_class_i);
    rng(rng_factor); idx_class_i_perm = randperm(num_class_i);
    idx_test_select_i = idx_class_i(idx_class_i_perm(1:num_test_class_i));
    idx_test_select = [idx_test_select,idx_test_select_i'];
end

end