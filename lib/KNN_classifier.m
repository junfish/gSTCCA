function estimate = KNN_classifier(vec_X_train, vec_X_test, label_train, k)
num_train = size(vec_X_train,1);
num_test = size(vec_X_test,1);
num_class = numel(unique(label_train));

for i = 1:num_test
    X_test_i = vec_X_test(i,:);
    for j = 1:num_train
        X_train_j = vec_X_train(j,:);
        distance(j) = norm(X_test_i - X_train_j);
    end
    
    if(k == 1)
        [~,min_idx] = min(distance);
        estimate_i = label_train(min_idx);
        estimate(i) = estimate_i;
    else
        [distance_sort, sort_idx] = sort(distance,'ascend');
        sort_idx_k = sort_idx(1:k);
        label_k = label_train(sort_idx_k);
        for c = 1:num_class
            num_c(c) = numel(find(label_k == c));
        end
        [~,label_num_sort] = sort(num_c,'descend');
        estimate(i) = label_num_sort(1);
    end
end

end