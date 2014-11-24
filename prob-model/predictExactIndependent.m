function p_0 = predictExactIndependent(X_next, mus, Sigmas, rho, pi)
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
      p_X_ks(k, i) = mvnpdff(X_next(:, i)', mus(:, k, i)', Sigmas(:, :, k, i))';
    endfor
  endfor
  %% XXX Workaround
  %if(sum(sum(p_X_ks, 1) < 10^-32) > 0)
  %  warning('Some services have zero probability for the given monitoring data. Assigning a low probability to obtain a prediction.')
  %  % Assign a small probability to all indices of the services with zero probability.
  %  % XXX Is 10^-8 a good value? Will it not be larger than some legit values in other services?
  %  p_X_ks(:, sum(p_X_ks, 1) < 10^-32) = 10^-8;
  %endif
  % Save the services with very low probabilites, to omit them in prediction.
  i_idx = sum(p_X_ks, 1) > 10^-32;
  % weigh the component probabilities with the mixture parameters
  p_X_ks = pi .* p_X_ks;
  p_d = 0;
  p_d_comp = 0;
  % Only iterate over the values where rho is not NaN.
  % TODO Vectorize and maybe run in parallel. For the experiment it is better to run the the outer loop in parallel where this function is called.
  for l = [1:K^I](!isnan(rho))
    % TODO Probably it would be way cheaper to increment Z_n directly instead of using the full decoding routine each time. This will only work as long as we iterate over all states and each state in sequence.
    [Z_n, z] = dec2oneofK(l, K, I);
    p_X_Z = prod(p_X_ks(Z_n(:, i_idx)));
    p_d = p_d + p_X_Z*rho(l)^d_next*(1 - rho(l))^(1 - d_next);
  endfor
  % normalize
  p_0 = prod(sum(p_X_ks(:, i_idx), 1))^-1 * p_d;
endfunction


function pdf = mvnpdff (x, mu = 0, sigma = 1)
  ## Check input
  if (!ismatrix (x))
    error ("mvnpdf: first input must be a matrix");
  endif
  
  if (!isvector (mu) && !isscalar (mu))
    error ("mvnpdf: second input must be a real scalar or vector");
  endif
  
  if (!ismatrix (sigma) || !issquare (sigma))
    error ("mvnpdf: third input must be a square matrix");
  endif

  [ps, ps] = size (sigma);
  [d, p] = size (x);
  if (p != ps)
    error ("mvnpdf: dimensions of data and covariance matrix does not match");
  endif
  
  if (numel (mu) != p && numel (mu) != 1)
    error ("mvnpdf: dimensions of data does not match dimensions of mean value");
  endif
  
  mu = mu (:).';
  if (all (size (mu) == [1, p]))
    mu = repmat (mu, [d, 1]);
  endif
  
  if (nargin < 3)
    pdf = (2*pi)^(-p/2) * exp (-sumsq (x-mu, 2)/2);
  else
    r = chol (sigma);
    pdf = (2*pi)^(-p/2);
    foo = (x-mu)/r;
    pdf = pdf * exp (-sum(foo .* foo, 2)/2);
    pdf = pdf / prod (diag (r));
  endif
endfunction

