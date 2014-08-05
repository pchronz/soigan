function p_Z = computePosteriorVectorized(mus, Sigmas, pi, rho, X, d, K)
  [D, D, K, I] = size(Sigmas);
  [D, I, N] = size(X);
  % The following allocate memory, that is needed in the l-loop. Putting it hear to avoid highly-frequent re-allocation.
  % select the right mus; needed in the l-loop;
  X_c = reshape(mat2cell(X, D, I, ones(1, N)), N);
  d_c = mat2cell(d, 1, ones(1, N))';
  p_Z = cell2mat(parcellfun(max(nproc() - 1, 1), createComputePosteriorN(D, K, I, mus, Sigmas, pi, rho), X_c, d_c, 'UniformOutput', false)');
  %for n = 1:N
  %  p_Z(:, n) = arrayfun(createComputePosteriorGlobal(D, K, I, mus, Sigmas, pi, X(:, :, n), d(n)), [1:K^I]', rho);
  %endfor
  % Scale the values, so that the largest un-normalized entry for the posterior for one n is 10.
  max_entries = max(p_Z);
  p_Z = p_Z - max_entries + log(10);
  assert(!any(any(isnan(p_Z))))
  % Un-log
  p_Z = e.^p_Z;
  assert(!any(any(isnan(p_Z))))
  % normalize
  %p_Z = p_Z ./ sum(p_Z);
  %p_Z = e.^(log(p_Z) .- log(sum(p_Z)));
  %for n = 1:N
  %  if(sum(p_Z(:, 1)) >= 1.00001 || sum(p_Z(:, 1)) <= 0.99999)
  %    p_Z(:, 1)
  %    n
  %    1 - sum(p_Z(:, 1))
  %    error('p_Z does not sum to one')
  %  endif
  %endfor
endfunction

function lp = logmvnpdf(x, mu, Sigma)
  D = length(x);
  % TODO Use Cholesky decomposition for better numerical stability?!
  lp = -0.5*D*log(2*pi) - 0.5*log(det(Sigma)) - 0.5*(x - mu)'*inv(Sigma)*(x - mu);
endfunction

function mu_l = selectmus(z_idx, mus_i)
  mu_l = mus_i(:, z_idx);
endfunction

function p_Z_n = computePosteriorGlobal(D, K, I, mus, Sigmas, pi, X_n, d_n, l, rho)
  [Z_n, z] = dec2oneOfK(l, K, I);
  % z_idx = (base2dec(z(1, :)', K)) + 1;
  for i = 1:I
    z_idx(i) = base2decfast(z(1, i), K) + 1;
  endfor
  % assert(z_idx == z_idx_fast)
  % Select the right mus
  % XXX The for loop seems to be way faster here.
  %mus_m = cell2mat(cellfun('selectmus', mat2cell(z_idx, ones(1, I)), reshape(mat2cell(mus, D, K, ones(1, I)), I, 1), 'UniformOutput', false)');
  for i = 1:I
    mus_l(:, i) = mus(:, z_idx(i), i);
  endfor
  % select the right Sigmas
  for i = 1:I
    Sigmas_l(:, :, i) = Sigmas(:, :, z_idx(i), i);
  endfor
  % select the right pis
  pi_l = pi(logical(Z_n));
  % compute the posterior for the current state and observation
  p_Z_n = log(rho^d_n) + log((1 - rho)^(1 - d_n));
  % DEBUG
  if(!isreal(p_Z_n))
    more on
    rho
    pi_l
    p_Z_n
    l
    more off
    error('p_Z(l, n) is not real')
  endif
  % Prepare inputs for vectorization.
  % XXX Not sure whether mat2cell is any faster than the for loop here.
  %X_c = cell(1, I);
  %mus_c = cell(1, I);
  %Sigmas_c = cell(1, I);
  %for i = 1:I
  %  X_c(i) = X(:, i, n);
  %  mus_c(i) = mus_l(:, i);
  %  Sigmas_c(i) = Sigmas_l(:, :, i);
  %endfor
  X_c = mat2cell(X_n, D, ones(1, I));
  mus_c = mat2cell(mus_l, D, ones(1, I));
  Sigmas_c = reshape(mat2cell(Sigmas_l, D, D, ones(1, I)), 1, I);
  log_p_x_n_is = cellfun('logmvnpdf', X_c, mus_c, Sigmas_c);
  p_Z_n = p_Z_n + sum(log(pi_l)) + sum(log_p_x_n_is);
  % DEBUG
  if(!isreal(p_Z_n))
    more on
    rho
    pi_l(i)
    log_p_x_n_is
    p_Z_n
    l
    more off
    error('p_Z(l, n) is not real')
  endif
endfunction

% Partial application to avoid repeating most arguments K^I times.
function f = createComputePosteriorGlobal(D, K, I, mus, Sigmas, pi, X_n, d_n)
  f = @(l, rho) computePosteriorGlobal(D, K, I, mus, Sigmas, pi, X_n, d_n, l, rho);
endfunction

% Partial application to avoid repeating most arguments K^I times.
function f = createComputePosteriorN(D, K, I, mus, Sigmas, pi, rho)
  f = @(X_n, d_n) arrayfun(createComputePosteriorGlobal(D, K, I, mus, Sigmas, pi, X_n, d_n), [1:K^I]', rho);
endfunction

