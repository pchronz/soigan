function [X_tr, d_tr, X_test, d_test] = splitDataRand(X, d, p)
  [D, I, N] = size(X);
  % split the data into training and validation sets
  % first split the data by target classes
  flags_0 = (d == 0);
  flags_1 = !flags_0;
  idx_0 = (1:N)(flags_0);
  idx_1 = (1:N)(flags_1);
  % 0s first
  assert(sum(flags_0) >= 2)
  idx_0 = idx_0(randperm(length(idx_0)));
  idx_0_tr = idx_0(1:ceil(p*length(idx_0)))
  idx_0_test = idx_0(ceil(p*length(idx_0)) + 1:end)
  % 1s second
  assert(sum(flags_1) >= 2)
  idx_1 = idx_1(randperm(length(idx_1)));
  idx_1_tr = idx_1(1:ceil(p*length(idx_1)))
  idx_1_test = idx_1(ceil(p*length(idx_1)) + 1:end)
  % now assemble both into common training and test sets
  assert(length(intersect(idx_1_tr, idx_1_test)) == 0)
  assert(length(intersect(idx_0_tr, idx_0_test)) == 0)
  X_test = X(:, :, [idx_0_test, idx_1_test]);
  d_test = d([idx_0_test, idx_1_test]);
  X_tr = X(:, :, [idx_0_tr, idx_1_tr]);
  d_tr = d([idx_0_tr, idx_1_tr]);
endfunction


