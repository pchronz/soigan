function Sigmas = maxSigmasSlow(X, mus, p_Z, D, K, I, N)
  % XXX Replace with pre-allocated Sigma if the allocation takes to long.
  Sigmas = zeros(D, D, K, I);
  %Sigmas = Sigmas .* 0;
  for i = 1:I
    for k = 1:K
      Sigma_norm = 0;
      for n = 1:N
        diff_n = X(:, :, n) - reshape(mus(:, k, :), D, I);
        p_Z_n = p_Z{n};
        idx_l = p_Z_n(1, :);
        for l = idx_l
          [Z_n, z] = dec2oneOfK(l, K, I);
          p_Z_n = p_Z{n};
          p_Z_nl = p_Z_n(2, p_Z_n(1, :) == l);
          Sigmas(:, :, k, i) += p_Z_nl * Z_n(k, i) * diff_n(:, i) * diff_n(:, i)';
          Sigma_norm = Sigma_norm + p_Z_nl * Z_n(k, i);
        endfor
      endfor
      Sigmas(:, :, k, i) = Sigmas(:, :, k, i) ./ Sigma_norm;
    endfor
  endfor
endfunction

