more off

pkg load statistics


% TODO predictive density
% TODO augment baseline model
% TODO sampling
% TODO variational inference

% create the data first
% D x I x N
N = 100;
% mixture components
I = 3;
D = 2;
X_1 = mvnrnd(0.7 * ones(1, D), 0.01 * eye(D) + 0 * rotdim(eye(D), 1), I*N)';
%X_1 = mvnrnd(0.7 * ones(1, D), 0.01 * eye(D) + 0 * rotdim(eye(D), 1), I*N*1/4)';
%X_2 = mvnrnd(0.3 * ones(1, D), 0.05 * eye(D) + 0 * rotdim(eye(D), 1), I*N*2/4)';
%X_3 = mvnrnd(0.1 * ones(1, D), 0.001 * eye(D) + 0 * rotdim(eye(D), 1), I*N*1/4)';
perm_vec = randperm(I*N);
%X = reshape([X_1, X_2, X_3](:, perm_vec), [D, N, I]);
X = reshape(X_1(:, perm_vec), [D, N, I]);
% mark the clusters after permutation
clusters = reshape((X_1(1, :) > 0.7)(perm_vec), [N, I]);
% d := 1, if at least two clusters are ones
d = (sum(clusters, 2) >= 2)';
X = rotdim(X, 1, [2, 3]);

% split the data into training and validation sets
idx = 0;
while(sum(idx) == 0)
  idx = unidrnd(10, 1, N);
  idx = idx > 8;
endwhile
X_test = X(:, :, idx);
d_test = d(idx);
X = X(:, :, not(idx));
d = d(not(idx));

disp('Baseline model training')
tic()
[centers, rho_base] = learnBaselineModel(2, X, d);
toc()

disp('Multi-mixture model training')
tic()
[mus, Sigmas, rho, pi] = learnExactIndependent(2, X, d, 10);
toc()

% evalute the Akaike information criterion
aic = computeAicIndependent(mus, Sigmas, rho, pi, X, d);

rho
mus
Sigmas
pi
aic

disp('Predictions tic')
tic()
N_test = size(X_test, 3);
hits = 0;
hits_baseline = 0;
for n = 1:N_test
  [p_0, p_1] = predictExactIndependent(X_test(:, :, n), mus, Sigmas, rho, pi);
  hits = hits + double((p_0 < p_1) == d_test(n));

  [p_0, p_1] = predictBaseline(X_test(:, :, n), centers, rho_base);
  hits_baseline = hits_baseline + double((p_0 < p_1) == d_test(n));
endfor
disp('Predictions toc')
toc()

disp('prob model');
hits / N_test

disp('baseline model');
hits_baseline / N_test





