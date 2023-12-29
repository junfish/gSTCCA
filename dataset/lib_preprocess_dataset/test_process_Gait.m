data_name = 'Gait17_64x44x20.mat';
load(data_name);
label = gnd;
X_origin = fea3D;
N = size(X_origin,ndims(X_origin));
%% optical-flow
for i = 1:N
% [U, V] = extract_optical_flow(video)
    Xi = X_origin(:,:,:,i);
    [Ui,Vi] = extract_optical_flow(Xi);
    U(:,:,:,i) = Ui; V(:,:,:,i) = Vi;
end

X_origin = single(X_origin); U = single(U); V = single(V);
