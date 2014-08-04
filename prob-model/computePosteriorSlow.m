function p_Z = computePosteriorSlow(mus, Sigmas, pi, rho, X, d, K)
    [D, D, K, I] = size(Sigmas);
    [D, I, N] = size(X);
    p_Z = zeros(K^I, N);
    % The following allocate memory, that is needed in the l-loop. Putting it hear to avoid highly-frequent re-allocation.
    % select the right mus; needed in the l-loop;
    mus_l = zeros(D, I);
    z_idx = zeros(I, 1);
    Sigmas_l = zeros(D, D, I);
    for n = 1:N
      if(mod(n, 50) == 0 || n == N)
        n
      endif
      %p_Z(:, n) = cellfun('computePosteriorGlobal', [1:K^I], mus, Sigmas, rho, pi, X(:, :, n), d(n));
      for l = 1:K^I
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
        p_Z(l, n) = log(rho(l)^d(n)) + log((1 - rho(l))^(1 - d(n)));
        % DEBUG
        if(!isreal(p_Z(l, n)))
          more on
          rho
          pi_l(i)
          log_p_x_n_i
          p_Z(l, n)
          l
          n
          i
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
        X_c = mat2cell(X(:, :, n), D, ones(1, I));
        mus_c = mat2cell(mus_l, D, ones(1, I));
        Sigmas_c = reshape(mat2cell(Sigmas_l, D, D, ones(1, I)), 1, I);
        log_p_x_n_is = cellfun('logmvnpdf', X_c, mus_c, Sigmas_c);
        p_Z(l, n) = p_Z(l, n) + sum(log(pi_l)) + sum(log_p_x_n_is);
        % DEBUG
        if(!isreal(p_Z(l, n)))
          more on
          rho
          pi_l(i)
          log_p_x_n_is
          p_Z(l, n)
          l
          n
          i
          more off
          error('p_Z(l, n) is not real')
        endif
      endfor
    endfor
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

