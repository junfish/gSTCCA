%% Add data path
cur_path = pwd;
cd('D:/Code/CCA');
addpath(genpath('Dataset'));
addpath(genpath(pwd))
cd(cur_path)
addpath('lib')
addpath('tensor_toolbox-v3.2.1')

%% Test other datasets

data_normalization = 0; Use_MPCA = 0; warning('off'); use_my_imple = 1; corrupted_ratio = 1;
% dataset_name = 'Gait17_32x22x10_processed'; [X,Y,label] = load_dataset(dataset_name, data_normalization);
% dataset_name = 'JAFFE'; mode = 'paired'; [X,Y,label] = load_JAFFE_dataset(data_normalization, mode);
dataset_name = 'Mnist_012_processed_new'; mode = 'normal';

disp(['#############################################################'])
disp(['testing: TCCA_youlin',' dataset: ',dataset_name,' ', 'normalization: ',num2str(data_normalization), ' Use_MPCA = ',num2str(Use_MPCA)])
disp(['#############################################################'])

ratio = 0.1;
count = 1;

corrupted_ratio = 1;  noise_variance = 0;

% [X,Y,label] = load_dataset_with_missing_data(dataset_name, data_normalization, corrupted_ratio); 
% disp(['Test influence of missing ratio',' noise level: ', num2str(noise_variance), ' missing_ratio: ',num2str(corrupted_ratio)])

[X,Y,label] = load_dataset_with_noise(dataset_name, data_normalization, corrupted_ratio, noise_variance);
disp(['Test influence of awgn noise',' noise level: ', num2str(noise_variance), ' corrupted_ratio: ',num2str(corrupted_ratio)])


