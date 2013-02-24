function [p_0, p_1] = predictFailure(X_next, mus, Sigmas, rho, pi)
  d_next = 0;
  % predict
  K = size(mus, 2);
  I = size(X_next, 2);
  % compute the probabilities for all of the components for the observation
  p_X = zeros(K, I);
  for k = 1:K
    p_X(k, :) = mvnpdf(X_next', mus(:, k)', Sigmas(:, :, k))';
  endfor
  % multiply the component densities by the mixture parameter
  p_X = diag(pi) * p_X;
  p_d = 0;
  for l = 1:K^I
    [Z_next, z] = dec2oneofK(l, K, I);
    p_d = p_d + rho(l)^d_next * (1 - rho(l))^(1-d_next) * prod(p_X(Z_next));
  endfor
  % normalize
  assert(size(sum(p_X, 1) == I))
  p_0 = (prod(sum(p_X, 1)))^-1 * p_d;

  % verify the implementation
  d_next = 1;
  p_d_comp = 0;
  for l = 1:K^I
    [Z_next, z] = dec2oneofK(l, K, I);
    p_d_comp = p_d_comp + rho(l)^d_next * (1 - rho(l))^(1-d_next) * prod(p_X(Z_next));
  endfor
  % normalize
  p_1 = (prod(sum(p_X, 1)))^-1 * p_d_comp;
  assert((p_0 + p_1) >= 0.99 && (p_0 + p_1) <= 1.01)
endfunction



