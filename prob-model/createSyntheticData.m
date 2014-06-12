function [X, d] = createSyntheticData()
  % create the data first
  % D x I x N
  N = 500;
  % mixture components
  I = 5;
  D = 3;
  K = 2;
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
endfunction

