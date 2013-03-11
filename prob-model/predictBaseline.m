function [p_0, p_1] = predictBaseline(X_next, centers, rho)
  [D, K] = size(centers);
  [D, I] = size(X_next);
  % identify the nearest cluster
  diffs = rotdim(centers(:, :, ones(1, I)), 1, [2, 3]) - X_next(:, :, ones(1, K));
  distances = zeros(I, K);
  clusters = zeros(1, I);
  for i = 1:I
    for k = 1:K
      distances(i, k) = diffs(:, i, k)' * diffs(:, i, k);
    endfor
    [d, idx] = min(distances(i, :));
    clusters(i) = idx;
  endfor

  % get the probability for the resulting cluster
  l = base2dec(num2str(clusters-1), K) + 1;
  p_1 = rho(l);
  p_0 = 1-p_1;
endfunction

