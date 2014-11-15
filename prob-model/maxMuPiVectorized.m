function [mus, pi] = maxMuPiVectorized(p_Z, X, K)
  global para;
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
      p_Z_c = p_Z;
      X_i_c = reshape(mat2cell(X(:, i, :), D, 1, ones(1, N)), N);
      %[m, p] = parcellfun(nproc(), @maxMuPiN, p_Z_c, X_i_c, 'UniformOutput', false, 'ErrorHandler', @(err) disp(err));
      if(para)
        [m, p] = parcellfun(nproc(), @maxMuPiN, p_Z_c, X_i_c, 'UniformOutput', false, 'ErrorHandler', @(err) disp(err));
      else
        [m, p] = cellfun(@maxMuPiN, p_Z_c, X_i_c, 'UniformOutput', false, 'ErrorHandler', @(err) disp(err));
      endif
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
  %for l = 1:Kay^I
  for l = p_Z_n(1, :)
    [Z_n, z] = dec2oneOfK(l, Kay, I);
    % pi
    pi = pi + Z_n(kay, iy) * p_Z_n(2, p_Z_n(1, :) == l);
    % mus
    mu = mu + p_Z_n(2, p_Z_n(1, :) == l) * Z_n(kay, iy) * x_i_n;
  endfor
  % XXX The loop is a lot faster than the closure. Why are closures so slow? Can they be avoided? How to pass in invariant arguments otherwise?
  %[pi_, mu_] = arrayfun(createMaxMuPiL(Kay, I, kay, iy, x_i_n), [1:Kay^I]', p_Z_n, 'UniformOutput', false);
  %
  %pi = sum(cell2mat(pi_));
  %mu = sum(cell2mat(mu_'), 2);
endfunction

function f = createMaxMuPiL(K, I, k, i, x_i_n)
  f = @(l, p_Z_n_l) maxMuPiL(K, I, k, i, x_i_n, l, p_Z_n_l);
endfunction

function [pi, mu] = maxMuPiL(K, I, k, i, x_i_n, l, p_Z_n_l)
  [Z_n, z] = dec2oneOfK(l, K, I);
  % pi
  pi = Z_n(k, i) * p_Z_n_l;
  % mus
  mu = p_Z_n_l * Z_n(k, i) * x_i_n;
endfunction


