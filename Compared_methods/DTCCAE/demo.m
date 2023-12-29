%% Dataset.
% Training data are named X1 and X2 for view 1/2.
% Tuning data are named XV1 and XV2 for view 1/2.
% Testing data are named XTe1 and XTe2 for view 1/2.
% All data matrices contains samples rowwise.
% 
% The dataset of randomly rotated + random background MNIST images used in 
%   the paper, can be generated using the script createMNIST.m and the mat 
%   file mnist_all.mat downloaded from Sam Roweis's webpage
%   http://www.cs.nyu.edu/~roweis/data.html
% 
% Uncomment the following line if you want to generate the dataset.
% createMNIST;

clear
load MNIST.mat

%% Fix the projection dimensionality to 10.
K=10;

%% Use the seed to reproduce the errors listed below.
randseed=8409;

%% Baseline: input features, takes long to classify.
[tuneerr,testerr]=svmclassify(X1,trainLabel,XV1,tuneLabel,XTe1,testLabel);
fprintf('Inputs: tune error rate=%.1f, test error rate=%.1f\n\n',...
  100*tuneerr,100*testerr);
%Inputs: tune error rate=14.6, test error rate=13.1
clear tuneerr testerr

%% Run linear CCA.
% Regularizations for each view.
rcov1=0.01; rcov2=100;
[P1,~,m1,~,~]=linCCA(X1,X2,K,rcov1,rcov2);
% Testing the linear CCA projections.
X1proj=bsxfun(@minus,X1,m1)*P1;
XV1proj=bsxfun(@minus,XV1,m1)*P1;
XTe1proj=bsxfun(@minus,XTe1,m1)*P1;
[tuneerr,testerr]=svmclassify(X1proj,trainLabel,XV1proj,tuneLabel,...
  XTe1proj,testLabel);
fprintf('Linear CCA: tune error rate=%.1f, test error rate=%.1f\n\n',...
  100*tuneerr,100*testerr);
%Linear CCA: tune error rate=20.9, test error rate=19.6
clear X1proj XV1proj XTe1proj tuneerr testerr

%% Run randomized kernel CCA.
% Regularizations for each view.
rcov1=1e-4; rcov2=1e-4;
% Number of random Fourier features to use.
M1=5000;  M2=5000;
% Kernel widths.
s1=25;  s2=56.25;
[W1,b1,~,~,P1,~,m1,~,~,~]=randKCCA(X1,X2,K,M1,s1,rcov1,M2,s2,rcov2,randseed);
% Testing the learned RKCCA projections.
X1proj=bsxfun(@minus,cos(bsxfun(@plus,X1*W1,b1)),m1)*P1;
XV1proj=bsxfun(@minus,cos(bsxfun(@plus,XV1*W1,b1)),m1)*P1;
XTe1proj=bsxfun(@minus,cos(bsxfun(@plus,XTe1*W1,b1)),m1)*P1;
[tuneerr,testerr]=svmclassify(X1proj,trainLabel,XV1proj,tuneLabel,...
  XTe1proj,testLabel);
fprintf('Randomized KCCA: tune error rate=%.1f, test error rate=%.1f\n\n',100*tuneerr,100*testerr);
%Randomized KCCA: tune error rate=6.9, test error rate=5.9
clear X1proj XV1proj XTe1proj tuneerr testerr

%% Hyperparameters for DCCA network architecture.
% Regularizations for each view.
rcov1=1e-4; rcov2=1e-4;
% Hidden activation type.
hiddentype='sigmoid';
% Architecture (hidden layer sizes) for view 1 neural network.
NN1=[1024 1024 1024 K];
% Architecture (hidden layer sizes)  for view 2 neural network.
NN2=[1024 1024 1024 K];
% Weight decay parameter.
l2penalty=1e-4;

%% Run DCCA with SGD. No pretraining is used.
% Minibatchsize.
batchsize=800;
% Learning rate.
eta0=0.01;
% Rate in which learning rate decays over iterations.
% 1 means constant learning rate.
decay=1;
% Momentum.
momentum=0.99;
% How many passes of the data you run SGD with.
maxepoch=50;
addpath ./deepnet/
[F1opt,~]=DCCA_train(X1,X2,XV1,XV2,[],[],K,hiddentype,NN1,NN2, ...
  rcov1,rcov2,l2penalty,batchsize,eta0,decay,momentum,maxepoch);
% Testing the learned networks.
X1proj=gather(deepnetfwd(X1,F1opt)); 
XV1proj=gather(deepnetfwd(XV1,F1opt));
XTe1proj=gather(deepnetfwd(XTe1,F1opt));
[tuneerr,testerr]=svmclassify(X1proj,trainLabel,XV1proj,tuneLabel,...
  XTe1proj,testLabel); %#ok<*SVMCLASSIFY>
fprintf('DCCA: tune error rate=%.1f, test error rate=%.1f\n\n',...
  100*tuneerr,100*testerr);
%DCCA: tune error rate=4.1, test error rate=3.8
clear X1proj XV1proj XTe1proj tuneerr testerr

%% Hyperparameters for DCCAE network architecture.
% Regularizations for each view.
rcov1=1e-4; rcov2=1e-4;
% Hidden activation type.
hiddentype='sigmoid';
outputtype='sigmoid';
% Architecture for view 1 feature extraction network.
NN1=[1024 1024 1024 K];
% Architecture for view 2 feature extraction network.
NN2=[1024 1024 1024 K];
% Architecture for view 1 reconstruction network.
NN3=[1024 1024 1024 784];
% Architecture for view 2 reconstruction network.
NN4=[1024 1024 1024 784];
% Weight decay parameter.
l2penalty=1e-4;

%% Run DCCAE with SGD.
% Reconstruction error term weight.
lambda=0.001;
% Minibatchsize for the correlation term.
cca_batchsize=800;
% Minibatchsize for reconstruction error term.
rec_batchsize=100;
% Learning rate.
eta0=0.01;
% Rate in which learning rate decays over iterations.
% 1 means constant learning rate.
decay=1;
% Momentum.
momentum=0.99;
% How many passes of the data you run SGD with.
maxepoch=14;
addpath ./deepnet/
% Pretraining is used to speedup autoencoder training.
pretrainnet='RBMPRETRAIN_K=10.mat';
[F1opt,~]=DCCAE_train( ...
  X1,X2,XV1,XV2,[],[],K,lambda,hiddentype,outputtype,...
  NN1,NN2,NN3,NN4,rcov1,rcov2,l2penalty, cca_batchsize,rec_batchsize,...
  eta0,decay,momentum,maxepoch,randseed,pretrainnet);
% Testing the learned networks.
X1proj=gather(deepnetfwd(X1,F1opt)); 
XV1proj=gather(deepnetfwd(XV1,F1opt));
XTe1proj=gather(deepnetfwd(XTe1,F1opt));
[tuneerr,testerr]=ClassificationSVM(X1proj,trainLabel,XV1proj,tuneLabel,...
  XTe1proj,testLabel);
fprintf('DCCAE: tune error rate=%.1f, test error rate=%.1f\n\n',...
  100*tuneerr,100*testerr);
%DCCAE: tune error rate=3.1, test error rate=2.8
clear X1proj XV1proj XTe1proj tuneerr testerr


