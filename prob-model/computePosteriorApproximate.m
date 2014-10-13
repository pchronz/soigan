function p_Z = computePosteriorApproximate(mus, Sigmas, pi, rho, X, d, K)
    warning('off', 'Octave:broadcast');
    % Tolerance values for min probs.
    % TODO Choose based on some rational reason.
    % TODO How do these numbers relate to probabilites?
    rho_tol = 0.01;
    x_tol_range = 1;
    x_tol_min = 0.1;
    pi_tol_range = 1;
    pi_tol_min = 0.01;
    % Inspect the dims.
    [D, D, K, I] = size(Sigmas);
    [D, I, N] = size(X);
    % TODO Change the format here.
    p_Z = -Inf*ones(K^I, N);
    % The following allocates memory, that is needed in the l-loop. Putting it hear to avoid highly-frequent re-allocation.
    % select the right mus; needed in the l-loop;
    mus_l = zeros(D, I);
    z_idx = zeros(I, 1);
    Sigmas_l = zeros(D, D, I);
    for n = 1:N
      if(mod(n, 50) == 0 || n == N)
        n
      endif
      % Compute the probabilities for all components.
      log_p_X_n = zeros(K, I);
      for k = 1:K
        for i = 1:I
          log_p_X_n(k, i) = logmvnpdf(X(:, i, n), mus(:, k, i), Sigmas(:, :, k, i));
        endfor
      endfor
      % Compute the probabilities for all relevant rhos.
      % TODO Get only the relevant rhos from the maximization and only use those.
      rhos_idx = 0;
      if(d(n) == 1)
        rhos_idx = [1:K^I](rho > rho_tol)';
      else
        rhos_idx = [1:K^I](rho < rho_tol)';
      endif
      % (idx, rhos) x amount of relevant rhos
      log_p_d_n = zeros(2, length(rhos_idx));
      log_p_d_n(1, :) = rhos_idx;
      log_p_d_n(2, :) = log(rho(rhos_idx));
      % To compute the relevant global states, combine p_X and p_d.
      % TODO Try AND or other operators.
      % TODO Try to choose the largest values.
      % TODO Ensure that at least one local state per service is active. In the worst case just choose the largest one for a service.
      %loc_states = and(and(log_p_X_n > (max(log_p_X_n) - log(x_tol_range)), log_p_X_n > log(x_tol_min)), and(pi > (max(pi)/pi_tol_range), pi > pi_tol_min));
      loc_states = and(log_p_X_n > log(x_tol_min), pi > pi_tol_min);
      % Ensure that at least one state per service is selected.
      for i = 1:I
        if(sum(loc_states(:, i)) == 0)
          [v_x, ix_x] = max(log_p_X_n(:, i));
          [v_pi, ix_pi] = max(pi(:, i));
          % Choose the index with the larger product of pi and p_x
          p_x = log_p_X_n(ix_x, i) + log(pi(ix_x, i));
          p_pi = log_p_X_n(ix_pi, i) + log(pi(ix_pi, i));
          ix = ix_pi;
          if(p_x > p_pi)
            ix = ix_x;
          endif
          loc_states(ix, i) = true;
        endif
      endfor
      assert(prod(sum(loc_states)) != 0)
      % Assemble the relevant states into global states.
      loc_states
      glob_states = assembleStates(loc_states);
      length(glob_states)
      % Add the global states from p_Z
      glob_states = union(glob_states, rhos_idx);
      disp(['#global states: ', num2str(length(glob_states)), ' K^I = ', num2str(K^I)])
      % TODO Return the relevant states and probs for X_n for maximization.
      % Compute the posterior for the relevant states.
      for l = glob_states
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
        % TODO Change the format of p_Z to contain idxs and the posterior.
        % TODO Change the format of rho to contain idxs and the parameter values.
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

function glob_states = assembleStates(loc_states)
  [K, I] = size(loc_states);
  % Allocate the target array.
  glob_states = zeros(prod(sum(double(loc_states)), 2), 1);
  % Assign numbers to the local states.
  loc_num = repmat([0:K - 1]', 1, I);
  % Compute the exponentials.
  % The encoding is most-significant digit first.
  for i = 1:I
    loc_num(:, i) = K^(I - i) .* loc_num(:, i);
  endfor
  % Select the indices based on loc_states 
  loc_cell = cell(I, 1);
  for i = 1:I
    loc_cell(i, 1) = loc_num(loc_states(:, i), i);
  endfor
  % Compute the relevant global states
  prev_blk_len = 1;
  for i = 1:I
    B = repmat(loc_cell{i}, 1, prev_blk_len);
    B = reshape(B', numel(B), 1);
    B = repmat(B, length(glob_states)/length(B), 1);
    glob_states += B;
    prev_blk_len *= length(loc_cell{i});
  endfor
  glob_states += 1;
  sz = size(glob_states);
  if(sz(1) > sz(2))
    glob_states = glob_states';
  endif
  assert(length(glob_states) == length(unique(glob_states)))
endfunction

