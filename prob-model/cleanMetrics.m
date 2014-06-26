function X_clean = cleanMetrics(X)
  % First remove the metrics with no variance at all.
  [N, D] = size(X);
  idx = var(X) == 0;
  X_cleen = X(:, !idx);
  % Then remove the colinear metrics; that is the metrics that lead to a singular covariance matrix.
  X_clean = zeros(N, 0);
  for d = 1:D
    X_clean(:, end + 1) = X(:, d);
    if(isdefinite(cov(X_clean)) != 1)
      X_clean(:, end) = [];
    endif
  end
  assert(isdefinite(cov(X_clean)) == 1)
endfunction

