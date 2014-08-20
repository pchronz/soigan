function [mus, pi] = maxMuPiVectorized(p_Z, X, K)
  global D
  global Kay
  Kay = K;
  global I
  [D, I, N] = size(X);
  pi = zeros(K, I);
  mus = zeros(D, K, I);
  for i = 1:I
    global iy
    iy = i;
    for k = 1:K
      global kay
      kay = k;
      % Prepare the data as cells
      p_Z_c = reshape(mat2cell(p_Z, K^I, ones(1, N)), N);
      X_i_c = reshape(mat2cell(X(:, i, :), D, 1, ones(1, N)), N);
      [m, p] = parcellfun(nproc(), @maxMuPiN, p_Z_c, X_i_c, 'UniformOutput', false);
      pi(k, i) = sum(cell2mat(p));
      mus(:, k, i) = sum(cell2mat(m'), 2);
      mus(:, k, i) = mus(:, k, i) ./ pi(k, i);
    endfor
  endfor
  pi = pi ./ N;
endfunction

function [mu, pi] = maxMuPiN(p_Z_n, x_i_n)
  global kay
  global iy
  global D
  global Kay
  global I

  mu = zeros(D, 1);
  pi = 0;
  for l = 1:Kay^I
    [Z_n, z] = dec2oneOfK(l, Kay, I);
    % pi
    pi = pi + Z_n(kay, iy) * p_Z_n(l);
    % mus
    mu = mu + p_Z_n(l) * Z_n(kay, iy) * x_i_n;
  endfor
endfunction


