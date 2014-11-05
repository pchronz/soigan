function Sigmas = maxSigmasVectorized(X, mus, p_Z, D, K, I, N)
  % XXX Replace with pre-allocated Sigma if the allocation takes to long.
  Sigmas = zeros(D, D, K, I);
  %Sigmas = Sigmas .* 0;
  for i = 1:I
    for k = 1:K
      X_c = reshape(mat2cell(X, D, I, ones(1, N)), N, 1);
      [Sigma_kin, Sigma_norm_n] = cellfun(createMaxSigmasN(mus(:, k, :), D, K, I, N, k, i), X_c, p_Z, 'UniformOutput', false);
      % Reduce Sigma_kin and Sigma_norm_n by summing for all n.
      Sigmas(:, :, k, i) = sum(reshape(cell2mat(Sigma_kin'), D, D, N), 3);
      Sigma_norm = sum(cell2mat(Sigma_norm_n));
      % If Sigma_norm == 0, we have not observed any data relevant for that component. Assign it a narrow variance. 
      % Would it rather make sense to give it a wide variance?
      if(Sigma_norm == 0)
        % TODO Use some more clever approximation for a minimal covariance.
        Sigmas(:, :, k, i) = eye(D);
      else
        Sigmas(:, :, k, i) = Sigmas(:, :, k, i) ./ Sigma_norm;
      endif
    endfor
  endfor
endfunction

function f = createMaxSigmasN(mus_k, D, K, I, N, k, i)
  f = @(X_n, p_Z_n, k_c, i_c) maxSigmasN(X_n, p_Z_n, mus_k, D, K, I, N, k, i);
endfunction

function [Sigma_kin, Sigma_norm_n] = maxSigmasN(X_n, p_Z_n, mus_k, D, K, I, N, k, i)
  Sigma_kin = zeros(D, D);
  Sigma_norm_n = 0;
  diff_n = X_n - reshape(mus_k, D, I);
  % TODO Intersect p_Z_n(1, :) with the states which have Z_n(k, i) == 1. That should save some more states.
  for l = p_Z_n(1, :)
    [Z_n, z] = dec2oneOfK(l, K, I);
    p_Z_nl = p_Z_n(2, p_Z_n(1, :) == l);
    Sigma_kin += p_Z_nl * Z_n(k, i) * diff_n(:, i) * diff_n(:, i)';
    Sigma_norm_n += p_Z_nl * Z_n(k, i);
  endfor
endfunction 

