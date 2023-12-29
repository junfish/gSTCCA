%% Load dataset and run gSTCCA
addpath(genpath(pwd))
data_normalization = 1; % data normalization
dataset_name = 'Mnist_012_processed_new'; [X,Y,label] = load_dataset(dataset_name, data_normalization);

Use_MPCA = 0; % Use MPCA to reduce dimensionality
epsilon = 0.15;  % Solve subproblem
tolerance = 0.01; % Solve subproblem

% Some hyperparameters 
max_iter = 10; % Max iteration (10 is often enough)
dim = 5; dim_x = dim; dim_y = dim; % rank
normalize = 0; % set to 1 when dealing with outliers
initialize = 'rand'; % Initialization
t_selected = 100; % Example solution path for subproblem

N = length(label); idx_all = 1:N;
Ndim_X = ndims(X); Ndim_Y = ndims(Y);

disp(['#############################################################'])
disp(['testing: gSTCCA',' dataset: ',dataset_name,' normalization: ',num2str(data_normalization), ' Use_MPCA = ',num2str(Use_MPCA)])
disp(['Specs: ',' normalize: ',num2str(normalize), ' tolerance = ',num2str(tolerance), ' dim = ',num2str(dim),' t_selected = ',num2str(t_selected), ' initialize = ',initialize ])
disp(['#############################################################'])

for ratio_test_per_class = 0.2
    iter = 1; acc_all_KNN = zeros(10,1); acc_all_SVM = zeros(10,1);
    for rng_factor = 000:111:999
        % data preprocessingc
        idx_test_select = select_test_ratio(label, ratio_test_per_class, rng_factor); % Select training/test data
        idx_train_select = setdiff(idx_all, idx_test_select);
        label_test = label(idx_test_select);
        label_train = label(idx_train_select);
        
        if(Ndim_X == 3)
            X_test = X(:,:,idx_test_select); X_train = X(:,:,idx_train_select);
            Y_test = Y(:,:,idx_test_select); Y_train = Y(:,:,idx_train_select);
            
            if(Use_MPCA == 1)
                Ux_MPCA = MPCA(X_train,0.98); Uy_MPCA = MPCA(Y_train,0.98);
                X_train = proj_2DCCA(X_train,Ux_MPCA{1},Ux_MPCA{2}); Y_train = proj_2DCCA(Y_train,Uy_MPCA{1},Uy_MPCA{2});
                X_test = proj_2DCCA(X_test,Ux_MPCA{1},Ux_MPCA{2}); Y_test = proj_2DCCA(Y_test,Uy_MPCA{1},Uy_MPCA{2});
            end
            
            tic
            [Ux,Uy, ~, ~, record_iter] = gSTCCA(X_train, Y_train, max_iter, dim_x, dim_y, epsilon, tolerance, normalize, initialize, t_selected);
            feature_extract_time(iter) = toc;
            
            X_train_proj = proj_2DCCA(X_train, Ux{1}, Ux{2}); Y_train_proj = proj_2DCCA(Y_train, Uy{1}, Uy{2});
            X_test_proj = proj_2DCCA(X_test,  Ux{1}, Ux{2}); Y_test_proj = proj_2DCCA(Y_test, Uy{1}, Uy{2});
            
            sparsity(iter) = (nnz(Ux{1}) + nnz(Ux{2}) + nnz(Uy{1}) + nnz(Uy{2}))/(numel(Ux{1}) + numel(Ux{2}) + numel(Uy{1}) + numel(Uy{2}));
            sparsity(iter) = 1 - sparsity(iter);
            
        elseif(Ndim_X == 4)
            
            X_test = X(:,:,:,idx_test_select); X_train = X(:,:,:,idx_train_select);
            Y_test = Y(:,:,:,idx_test_select); Y_train = Y(:,:,:,idx_train_select);
            
            if(Use_MPCA == 1)
                Ux_MPCA = MPCA(X_train,0.98); Uy_MPCA = MPCA(Y_train,0.98);
                X_train = proj_3DCCA(X_train,Ux_MPCA); Y_train = proj_3DCCA(Y_train,Uy_MPCA);
                X_test = proj_3DCCA(X_test,Ux_MPCA); Y_test = proj_3DCCA(Y_test,Uy_MPCA);
            end
            
            tic
            [Ux,Uy, ~, ~, record_iter] = gSTCCA(X_train, Y_train, max_iter, dim_x, dim_y, epsilon, tolerance, normalize, initialize, t_selected);
            feature_extract_time(iter) = toc;
            
            X_train_proj = proj_3DCCA(X_train,Ux); Y_train_proj = proj_3DCCA(Y_train,Uy);
            X_test_proj = proj_3DCCA(X_test,Ux); Y_test_proj = proj_3DCCA(Y_test,Uy);
            
            sparsity(iter) = (nnz(Ux{1}) + nnz(Ux{2}) + nnz(Ux{3}) + nnz(Uy{1}) + nnz(Uy{2}) + nnz(Uy{3}))/(numel(Ux{1}) + numel(Ux{2}) + numel(Ux{3}) + numel(Uy{1}) + numel(Uy{2}) + numel(Uy{3}));
            sparsity(iter) = 1 - sparsity(iter);
            
        end
        
        vec_X_train_proj = tenmat(X_train_proj,ndims(X)); vec_Y_train_proj = tenmat(Y_train_proj,ndims(Y));
        vec_X_test_proj = tenmat(X_test_proj,ndims(X)); vec_Y_test_proj = tenmat(Y_test_proj,ndims(Y));
        
        vec_X_train_proj = double(vec_X_train_proj); vec_Y_train_proj = double(vec_Y_train_proj);
        vec_X_test_proj = double(vec_X_test_proj); vec_Y_test_proj = double(vec_Y_test_proj);
        
        Final_train_feature = [vec_X_train_proj,vec_Y_train_proj];
        Final_test_feature = [vec_X_test_proj,vec_Y_test_proj];
        
        % Classify
        % Using Nearest neighbour classifier
        knn = 1; 
        estimate = KNN_classifier(Final_train_feature, Final_test_feature, label_train, knn);
        acc_KNN = cal_acc(estimate, label_test);
        
        % Using multiclass SVM
        svmModel = fitcecoc(Final_train_feature,label_train);
        [estimate,SCORE] = predict(svmModel, Final_test_feature);
        acc_SVM = cal_acc(estimate, label_test);
        
        disp(['iter  = ',num2str(iter),' acc_KNN = ',num2str(acc_KNN), ' acc_SVM = ', num2str(acc_SVM)]);
        acc_all_KNN(iter) = acc_KNN; acc_all_SVM(iter) = acc_SVM;
        iter = iter + 1;
        
    end
        
    avg_feature_extract_time = mean(feature_extract_time); std_feature_extract_time = std(feature_extract_time);
    
    disp(['testing: gSTCCA',' dataset: ',dataset_name])
    disp(['ratio_test_per_class = ',num2str(ratio_test_per_class)])
    disp(['avg_feature_extract_time = ',num2str(avg_feature_extract_time), ' std_feature_extract_time = ',num2str(std_feature_extract_time)])
    disp(['avg_sparsity = ', num2str(mean(sparsity)), ' std_sparsity = ', num2str(std(sparsity))])
    disp([' acc_KNN = ',num2str(mean(acc_all_KNN)), ' acc_SVM = ', num2str(mean(acc_all_SVM))])
    disp([' std_KNN = ',num2str(std(acc_all_KNN)), ' std_SVM = ', num2str(std(acc_all_SVM))])
    disp('********************************************************************************')
    
end
