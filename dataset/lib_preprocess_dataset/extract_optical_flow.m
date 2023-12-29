function [U, V] = extract_optical_flow(video)

n_dim = ndims(video);
if(n_dim == 3)
    [U,V] = cal_gray_scale_optical_flow(video);
else
    [U,V] = cal_color_optical_flow(video);
end

end

function [U, V] =  cal_gray_scale_optical_flow(video)

opticFlow = opticalFlowHS;
[H, W, N] = size(video);
U = zeros(H,W,N); V = U;

for i = 1:N
    frameGray = video(:,:,i);
    flow = estimateFlow(opticFlow,frameGray);
    U(:,:,i) = flow.Vx;
    V(:,:,i) = flow.Vy;
end

end

function [U, V] =  cal_color_optical_flow(video)

addpath('C:\Users\KZM11\Documents\MATLAB\Code\CCA_related\RLOFLib-master\Matlab');

parameter = struct('maxLevel', 5, 'maxIter', 30, 'HampelNormS0', 3.2, 'HampelNormS1', 7, ...
    'LargeWinSize', 21 , 'SmallWinSize', 9, 'MinEigenvalue', 0.001, 'RansacReprojThresholdPercentil' , 71, ...
    'SegmentationThreshold', 30, 'useIlluminationModel' , 1, 'useGlobalMotionPrior', 1, ...
    'SupportRegionType', 'SR_FIXED', 'SolverType', 'ST_BILINEAR', ...
    'options', 'PrintParameter' );

[H, W, ~, N] = size(video);
U = zeros(H,W,3,N-1); V = U;

for i = 2:N-1
    image1 = video(:,:,:,i - 1);
    image2 = video(:,:,:,i);
    image1 = uint8(image1); image2 = uint8(image2);
    
    [Ui, Vi] = mex_DenseRLOF(image1, image2, parameter);
    U(:,:,:,i-1) = Ui;
    V(:,:,:,i-1) = Vi;
end

end