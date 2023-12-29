
%% Data generation.
clear all;
load mnist_origin;
TRAIN = {train1,train2,train3,train4,train5,train6,train7,train8,train9,train0};
TEST = {test1,test2,test3,test4,test5,test6,test7,test8,test9,test0};

% Create traning set.
X1=[];  X2=[];  trainLabel=[];
XV1=[]; XV2=[]; tuneLabel=[];

NUMVALID=1000;
load demoseed.mat randseed;
rng(randseed);

for i=1:10
  TMP = TRAIN{i};
  rp1 = randperm(size(TMP,1));
  rp2 = randperm(size(TMP,1));
  for j=1:size(TMP,1)
    tmp1 = double(TMP(rp1(j),:))/255; tmp1 = reshape(tmp1,28,28)';
    % Random rotation.
    angle = sign(randn)*rand*45;
    tmp1 = imrotate(tmp1,angle,'bicubic','crop');
    tmp1(tmp1>1)=1; tmp1(tmp1<0)=0;
    % Random background.
    tmp2 = double(TMP(rp2(j),:))/255; tmp2 = reshape(tmp2,28,28)';
    tmp2 = rand(28,28)+tmp2;
    tmp2(tmp2>1)=1; tmp2(tmp2<0)=0;
    if j<=NUMVALID
      XV1=[XV1; tmp1(:)'];
      XV2=[XV2; tmp2(:)'];
      tuneLabel=[tuneLabel; i];
    else
      X1=[X1; tmp1(:)'];
      X2=[X2; tmp2(:)'];
      trainLabel=[trainLabel; i];
    end
  end
end

XTe1=[];XTe2=[];testLabel=[];
for i=1:10
  TMP = TEST{i};
  rp1 = randperm(size(TMP,1));
  rp2 = randperm(size(TMP,1));
  for j=1:size(TMP,1)
    tmp1 = double(TMP(rp1(j),:))/255; tmp1 = reshape(tmp1,28,28)';
    % Random rotation.
    angle = sign(randn)*rand*45;
    tmp1 = imrotate(tmp1,angle,'bicubic','crop');
    tmp1(tmp1>1)=1; tmp1(tmp1<0)=0;
    % Random background.
    tmp2 = double(TMP(rp2(j),:))/255; tmp2 = reshape(tmp2,28,28)';
    tmp2 = rand(28,28)+tmp2;
    tmp2(tmp2>1)=1; tmp2(tmp2<0)=0;
    XTe1=[XTe1; tmp1(:)'];
    XTe2=[XTe2; tmp2(:)'];
    testLabel=[testLabel; i];
  end
end

save MNIST.mat X1 X2 trainLabel XV1 XV2 tuneLabel XTe1 XTe2 testLabel NUMVALID randseed
