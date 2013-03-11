function [centers, rho] = learnBaselineModel(K, X, d)
  [D, I, N] = size(X);
  % first do a clustering on the whole monitoring data
  % run k-means to obtain an initial estimate for the mixture components (mean and covariance)
  Xk = reshape(X, D, I*N, []);
  [idx, centers] = kmeans(Xk', K);
  centers = centers';
  % reshape the idx vector to conform with X's format (1xIxN)
  states = reshape(idx', I, N);
  rho = 0.5 * ones(K^I, 1);
  for l = 1:K^I
    % XXX this will only work for 1 < K < 10
    l_state = str2num(dec2base(l-1, K, I)') + 1;
    l_observed = prod(states == l_state(:, ones(1, N)));
    if(sum(l_observed) > 0)
      rho(l) = sum(d(logical(l_observed))) / sum(l_observed);
    endif
  endfor
endfunction

