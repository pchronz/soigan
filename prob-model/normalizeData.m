function nrmlzd = normalizeData(X)
  [N, D] = size(X);
  assert(min(eig(cov(X))) > 0)
  s = var(X);
  normalized = X(:, s > 0);
  normalized = normalized - mean(normalized)(ones(1, N), :);
  normalized = normalized * diag(1 ./ sqrt(var(normalized)));
  assert(sum(abs(mean(normalized))) < 0.0001);
  s2 = sum(var(normalized)) / size(normalized)(2);
  assert(s2 < 1.0001 && s2 > 0.9999);
  % Purge the data set of colinear variables. Otherwise the data set may have a singular covariance.
  nrmlzd = zeros(N, 0);
  for d = 1:D
    nrmlzd(:, end + 1) = normalized(:, d);
    if(min(eig(cov(nrmlzd))) <= 0)
      nrmlzd(:, end) = [];
    endif
  end
  assert(min(eig(cov(nrmlzd))) > 0)
endfunction


