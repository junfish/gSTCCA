%% Generate data: 1D X vs 2D Y
clear;

% 1D X true signal: 100-by-1 vector
V = zeros(100, 1);
V([13 24 57 73 92]) = 1;

% 2D Y true signal: 64-by-64 cross
%shapename = 'rectangle';
shapename = 'cross';
%shapename = 'disk';
%shapename = 'butterfly';
system(['rm -rf ./' shapename]);
system(['mkdir ./' shapename]);
system(['cp ' shapename '.gif' ' ./' shapename]);
shape = imread([shapename '.gif']);
shape = array_resize(shape,[32,32]); % 32-by-32
W = zeros(2*size(shape));
W((size(W,1)/4):(size(W,1)/4)+size(shape,1)-1, ...
    (size(W,2)/4):(size(W,2)/4)+size(shape,2)-1) = shape;
[p1,p2] = size(W);

figure;
set(gca, 'FontSize', 15);
subplot(1, 2, 1)
hold on;
plot(double(V), '+')
title('True V');
axis square;
subplot(1, 2, 2)
imagesc(W);
colormap(gray);
title('True W');
axis square;
% saveas(gcf, ['./' shapename '/' shapename '_true'], 'epsc');

% simulate joint normal random deviates
n = 1000;
rho = 0.95;
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
[X, Y] = simcca(V, W(:), rho, n, 'noisex', 1, 'noisey', 1e-3);
display(corr(X * V, Y * W(:))); % sample correlation

%% Classical CCA with ridge

% S = cov([X Y], 1);
% px = size(X, 2);
% py = size(Y, 2);
% B = sparse([S(1:px, 1:px) zeros(px, py); zeros(py, px) S(px+1:end, px+1:end)]);
% % add ridge term
% B = B + 1e-3 * speye(size(B, 1));
% A = sparse([zeros(px, px) S(1:px, px+1:end); S(px+1:end, 1:px) zeros(py, py)]);
% [evec, eval] = eigs(A, B, 1, 'lm');
% rho = corr(X * evec(1:px), Y * evec(px+1:end));
% betax = sign(rho) * evec(1:px);
% betay = reshape(evec(px+1:end), 64, 64);
% rho = abs(rho);
% 
% figure;
% set(gca, 'FontSize', 15);
% subplot(1, 2, 1)
% hold on;
% plot(abs(betax), 'o')
% title({'Classical CCA + ridge'; ['\angle (V, Vhat) = ' ...
%   num2str(abs(sum((V(:) / norm(V(:))) .* (betax / norm(betax)))))]});
% axis square;
% subplot(1, 2, 2)
% imagesc(betay);
% colormap(gray);
% title({['\rhohat = ' num2str(rho)]; ['\angle (W, What) = ' ...
%   num2str(abs(sum((W(:) / norm(W(:))) .* (betay(:) / norm(betay(:))))))]});
% axis square;
% saveas(gcf, ['./' shapename '/' shapename '_ccaridge'], 'epsc');

%% Tensor CCA

% make data into tensors
if ~isa(X, 'tensor');
  X = tensor(X');
  Y = tensor(Y', [p1 p2 n]);
end

% fit rank-(1, 1) tensor CCA
rx = 1;
lambda = 0;

for ry = 1:3
  
[betax, betay, rho] = tcca(X, Y, rx, ry, lambda);

figure;
set(gca, 'FontSize', 15);
subplot(1, 2, 1)
hold on;
plot(abs(double(betax)), 'o')
title({['TCCA, rank = (', num2str(rx) ', ' num2str(ry) ')']; ...
  ['\angle (V, Vhat) = ' num2str(abs(sum((V(:) / norm(V(:))) ...
  .* (double(betax(:)) / norm(double(betax(:)))))))]});
axis square;
subplot(1, 2, 2)
imagesc(double(full(betay)));
colormap(gray);
title({['\rhohat = ' num2str(rho)]; ...
  ['\angle (W, What) = ' num2str(abs(sum((W(:) / norm(W(:))) ...
  .* (vec(double(full(betay))) / norm(betay)))))]});
axis square
% saveas(gcf, ['./' shapename '/' shapename '_tcca_' num2str(rx) '_' num2str(ry)], 'epsc');

end

%% Sparse tensor CCA

% make data into tensors
% if ~isa(X, 'tensor');
%   X = tensor(X');
%   Y = tensor(Y', [p1 p2 n]);
% end
% 
% % fit rank-(1, 1) tensor CCA
% rx = 1;
% lambda = 0;
% 
% for ry = 1
%   
% [betax, betay, rho] = tcca(X, Y, rx, ry, lambda, 'xl0max', 5, ...
%   'yl0max', floor(64*ry/3));
% 
% figure;
% set(gca, 'FontSize', 15);
% subplot(1, 2, 1)
% hold on;
% plot(abs(double(betax)), 'o')
% title({['SpTCCA, rank = (', num2str(rx) ', ' num2str(ry) ')']; ...
%   ['\angle (V, Vhat) = ' num2str(abs(sum((V(:) / norm(V(:))) ...
%   .* (double(betax(:)) / norm(double(betax(:)))))))]});
% axis square;
% subplot(1, 2, 2)
% imagesc(double(full(betay)));
% colormap(gray);
% title({['\rhohat = ' num2str(rho)]; ...
%   ['\angle (W, What) = ' num2str(abs(sum((W(:) / norm(W(:))) ...
%   .* (vec(double(full(betay))) / norm(betay)))))]});
% axis square
% saveas(gcf, ['./' shapename '/' shapename '_sptcca_' ...
%   num2str(rx) '_' num2str(ry)], 'epsc');
% 
% end

%% Tensor CCA assuming separable covariance structure

% fit rank-(1, 1) tensor CCA
% rx = 1;
% lambda = 0;
% 
% for ry = 1
%   
% [betax, betay, rho] = tcca(X, Y, rx, ry, lambda, 'covstr', 'sep');
% 
% figure;
% set(gca, 'FontSize', 15);
% subplot(1, 2, 1)
% hold on;
% plot(abs(double(betax)), 'o')
% title({['TCCA sep. cov., rank = (', num2str(rx) ', ' num2str(ry) ')']; ...
%   ['\angle (V, Vhat) = ' num2str(abs(sum((V(:) / norm(V(:))) ...
%   .* (double(betax(:)) / norm(double(betax(:)))))))]});
% axis square;
% subplot(1, 2, 2)
% imagesc(double(full(betay)));
% colormap(gray);
% title({['\rhohat = ' num2str(rho)]; ...
%   ['\angle (W, What) = ' num2str(abs(sum((W(:) / norm(W(:))) ...
%   .* (vec(double(full(betay))) / norm(betay)))))]});
% axis square
% saveas(gcf, ['./' shapename '/' shapename '_tcca_sepcov_' ...
%   num2str(rx) '_' num2str(ry)], 'epsc');
% 
% end

%% Sparse tensor CCA assuming separable covariance

% make data into tensors
% if ~isa(X, 'tensor');
%   X = tensor(X');
%   Y = tensor(Y', [p1 p2 n]);
% end
% 
% % fit rank-(1, 1) tensor CCA
% rx = 1;
% lambda = 0;
% 
% for ry = 1
%   
% [betax, betay, rho] = tcca(X, Y, rx, ry, lambda, 'covstr', 'sep', ...
%   'xl0max', 5, 'yl0max', floor(64*ry/3));
% 
% figure;
% set(gca, 'FontSize', 15);
% subplot(1, 2, 1)
% hold on;
% plot(abs(double(betax)), 'o')
% title({['SpTCCA sep. cov., rank = (', num2str(rx) ', ' num2str(ry) ')']; ...
%   ['\angle (V, Vhat) = ' num2str(abs(sum((V(:) / norm(V(:))) ...
%   .* (double(betax(:)) / norm(double(betax(:)))))))]});
% axis square;
% subplot(1, 2, 2)
% imagesc(double(full(betay)));
% colormap(gray);
% title({['\rhohat = ' num2str(rho)]; ...
%   ['\angle (W, What) = ' num2str(abs(sum((W(:) / norm(W(:))) ...
%   .* (vec(double(full(betay))) / norm(betay)))))]});
% axis square
% saveas(gcf, ['./' shapename '/' shapename '_sptcca_sepcov_' ...
%   num2str(rx) '_' num2str(ry)], 'epsc');
% 
% end
