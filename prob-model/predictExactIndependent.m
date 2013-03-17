function [p_0, p_1] = predictExactIndependent(X_next, mus, Sigmas, rho, pi)
  d_next = 0;
  % predict
  [D, K, I] = size(mus);
  % compute the component probabilities, all of which we will need
  p_X_ks = zeros(K, I);
  for k = 1:K
    for i = 1:I
      p_X_ks(k, i) = mvnpdf(X_next(:, i)', mus(:, k, i)', Sigmas(:, :, k, i))';
    endfor
  endfor
  % weigh the component probabilities with the mixture parameters
  p_X_ks = pi .* p_X_ks;
  p_d = 0;
  for l = 1:K^I
    [Z_n, z] = dec2oneofK(l, K, I);
    p_X_Z = prod(p_X_ks(Z_n));
    p_d = p_d + p_X_Z * rho(l)^d_next * (1-rho(l))^(1-d_next);
  endfor
  % normalize
  p_0 = prod(sum(p_X_ks, 1))^-1 * p_d;

  % verify the implementation
  d_next = 1;
  p_d_comp = 0;
  for l = 1:K^I
    [Z_n, z] = dec2oneofK(l, K, I);
    p_X_Z = prod(p_X_ks(Z_n));
    p_d_comp = p_d_comp + p_X_Z * rho(l)^d_next * (1-rho(l))^(1-d_next);
  endfor
  % normalize
  p_1 = prod(sum(p_X_ks, 1))^-1 * p_d_comp;
  assert((p_0 + p_1) >= 0.99 && (p_0 + p_1) <= 1.01)
endfunction



