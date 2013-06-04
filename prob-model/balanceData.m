function [Xbal, dbal] = balanceData(X, d)
  % determine which result appears less often
  t = sum(d == 1) < sum(d == 0);
  t_idx = d == t;
  % get the according observations
  X_t = X(:, :, t_idx);
  deltaN = sum(d == !t) - sum(d == t);
  [D, I, N_t] = size(X_t);
  ran_idx = randi(1:N_t, 1, deltaN);
  X_t_new = X_t(:, :, ran_idx);
  d_t_new = double(t) * ones(1, deltaN);
  Xbal = cat(3, X, X_t_new);
  dbal = [d, d_t_new];
endfunction

