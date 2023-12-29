function [tuneerr,testerr]=svmclassify...
  (X_train,Label_train,X_tune,Label_tune,X_test,Label_test)
% Wrapper for libsvm's linear one-vs-one classifier.
% Searches penalty parameter for hinge loss on a small grid.
% X_train/X_tune/X_test contain samples rowwise.

Label_train=Label_train(:);
Label_tune=Label_tune(:);
Label_test=Label_test(:);

addpath ./libsvm-master/;
addpath ./libsvm-master/matlab/;

CC=[1e-2 1e-1 1 1e1];
tuneerr=inf;

for j=1:length(CC)
  c=CC(j);
  options=['-q -s 0 -t 0 -c ' num2str(c)];
  
  model=svmtrain(Label_train,X_train,options);
  pred=svmpredict(Label_tune,X_tune,model);
  tmperr=sum(pred~=Label_tune)/length(Label_tune);
  
  if tmperr<tuneerr
    tuneerr=tmperr;
    optmodel=model;
  end
  
end
pred=svmpredict(Label_test,X_test,optmodel);
testerr=sum(pred~=Label_test)/length(Label_test);
