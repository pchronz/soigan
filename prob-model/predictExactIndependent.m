function [p_0, p_1] = predictExactIndependent(X_next, mus, Sigmas, rho, pi)
  d_next = 0;
  % predict
  [D, K, I] = size(mus);
  % DEBUG
  % Make sure that there are no singular covariance matrices.
  for k = 1:K
    for i = 1:I
      Sigma_ki = Sigmas(:, :, k, i);
      assert(isdefinite(Sigma_ki) == 1)
    endfor
  endfor
  % compute the component probabilities, all of which we will need
  p_X_ks = zeros(K, I);
  for k = 1:K
    for i = 1:I
      p_X_ks(k, i) = mvnpdf(X_next(:, i)', mus(:, k, i)', Sigmas(:, :, k, i))';
    endfor
  endfor
  % XXX Workaround
  if(sum(sum(p_X_ks, 1) < 10^-32) > 0)
    warning('Some services have zero probability for the given monitoring data. Assigning a low probability to obtain a prediction.')
    % Assign a small probability to all indices of the services with zero probability.
    p_X_ks(:, sum(p_X_ks, 1) < 10^-32) = 10^-8;
  endif
  % weigh the component probabilities with the mixture parameters
  p_X_ks = pi .* p_X_ks;
  p_d = 0;
  for l = 1:K^I
    [Z_n, z] = dec2oneofK(l, K, I);
    p_X_Z = prod(p_X_ks(Z_n));
    p_d = p_d + p_X_Z*rho(l)^d_next*(1 - rho(l))^(1 - d_next);
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
  % DEBUG
  if(!((p_0 + p_1) >= 0.99 && (p_0 + p_1) <= 1.01))
    p_0
    p_1
    p_0 + p_1
    p_d
    p_X_ks
    mus
    Sigmas
    rho
    X_next
    p_X_Z
    %Xn = X_next(:, 4);
    %mun = mus(:, :, 4);
    %Sigman = Sigmas(:, :, :, 4);
    %save temp.mat Xn mun Sigman
    error('The predicted values in the full probabilistic model do not sum to one.')
  endif
endfunction



