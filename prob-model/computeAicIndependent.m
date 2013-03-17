function aic = computeAicIndependent(mus, Sigmas, rho, pi, X, d)
  % evaluate the AIC
  [D, I, N] = size(X);
  K = size(mus, 2);
  % compute the log marginal likelihood first
  log_p_ns = zeros(1, N);
  for n = 1:N
    % compute the component probabilities, all of which we will need
    p_X_n_ks = zeros(K, I);
    for k = 1:K
      for i = 1:I
        p_X_n_ks(k, i) = mvnpdf(X(:, i, n)', mus(:, k, i)', Sigmas(:, :, k, i))';
      endfor
    endfor
    % weigh the component probabilities with the mixture parameters
    p_X_n_ks = pi .* p_X_n_ks;
    p_X_Z = 0;
    for l = 1:K^I
      [Z_n, z] = dec2oneofK(l, K, I);
      p_X_Z = p_X_Z + prod(p_X_n_ks(Z_n)) * rho(l)^d(n) * (1-rho(l))^(1-d(n));
    endfor
    log_p_ns(1, n) = log(p_X_Z);
  endfor
  log_p = sum(log_p_ns);

  aic = log_p - (K^I + K * I + I * K * 0.5 * (D^2 + D));
endfunction

