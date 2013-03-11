function [mus, Sigmas, rho, pi] = learnFailurePredictor(K, X, d, max_iter)
  disp('Init tic')
  tic()
  [D, I, N] = size(X);
  % starting the learning phase
  % allocate space for the model parameters
  mus = zeros(D, K);
  Sigmas = zeros(D)(:, :, ones(1, K));
  pi = 1/K * ones(K, 1);
  rho = 0.5 * ones(K^I, 1);

  % run k-means to obtain an initial estimate for the mixture components (mean and covariance)
  Xk = reshape(X, D, I*N, []);
  [idx, centers] = kmeans(Xk', K);
  mus = centers';
  % compute the intra-cluster covariance and use it to initialize the component covariance
  for k = 1:K
    Sigmas(:, :, k) = cov(Xk(:, idx' == k)');
  endfor

  % keep a copy of the previous model params to monitor the progress
  mus_prev = mus;
  Sigmas_prev = Sigmas;
  pi_prev = pi;
  rho_prev = rho;

  disp('Init toc')
  toc()

  for it = 1:max_iter
    disp('E-step tic');
    tic()
    it
    % save the current values before updating the params in this iteration
    mus_prev = mus;
    Sigmas_prev = Sigmas;
    pi_prev = pi;
    rho_prev = rho;

    % E-step
    % compute the posterior for all possible states for Z for all observations N
    % this results in a matrix with (K^I)*N entries, you might want to optimize this at some point
    % nevertheless we will need the posterior many times, so it should help us quite a lot
    % let's get it over with...
    p_Z = zeros(K^I, N);
    for n = 1:N
      % iteration of the states in Z_n
      for l = 1:K^I
        % transform the decimal number into the matrix Z_n with column vectors coded as 1-of-K
        [Z_n, z] = dec2oneofK(l, K, I);
        p_d_n = rho(l)^d(n) * (1-rho(l))^(1-d(n));
        p_X_n = 1;
        for i = 1:I
          % select the right mixture component for this state vector
          mu_k = mus(:, str2num(z(1, i)) + 1);
          Sigma_k = Sigmas(:, :, str2num(z(1, i)) + 1);
          % compute the likelihood for the system observations and multiply by the total likelihood for all systems
          % XXX fix to multivariate normal
          % XXX what to do if this value becomes small beyond machine precision? This might happen when I becomes large.
          % ==> probably then this value does not matter anyway and we keep it at zero.
          p_X_n = p_X_n * mvnpdf(X(:, i, n)', mu_k', Sigma_k);
        endfor
        % p_Z_n now
        % select the right components
        Pi = pi(:, ones(1, I));
        pi_l = Pi(Z_n);
        p_Z_n = prod(pi_l);
        p_Z(l, n) = p_d_n * p_X_n * p_Z_n;
      endfor
    endfor
    % normalize the values
    p_Z = p_Z ./ sum(p_Z)(ones(K^I, 1), :);
    gotnan = sum(sum(isnan(p_Z)));
    if(gotnan) 
      disp('Singularity!');
      break;
    endif
    disp('E-step toc')
    toc()

    disp('M-step tic')
    tic()
    % M-step
    % rho
    for l = 1:K^I
      rho(l) = sum(p_Z(l, :) * d') / sum(p_Z(l, :));
    endfor
    % pi
    for k = 1:K
      pi(k) = 0;
      for n = 1:N
        for l = 1:K^I
          [Z_n, z] = dec2oneofK(l, K, I);
          pi(k) = pi(k) + p_Z(l, n) * sum(Z_n(k, :), 2);
        endfor
      endfor
      pi(k) = 1/(I*N) * pi(k);
    endfor
    % mu
    for k = 1:K
      mus(:, k) = zeros(D, 1);
      for n = 1:N
        for l = 1:K^I
          [Z_n, z] = dec2oneofK(l, K, I);
          mus(:, k) = mus(:, k) + p_Z(l, n) * sum(X(:, :, n) * diag(Z_n(k, :)), 2);
        endfor
      endfor
      % normalize
      C = 0;
      for n = 1:N
        for l = 1:K^I
          [Z_n, z] = dec2oneofK(l, K, I);
          C = C + p_Z(l, n) * sum(Z_n(k, :));
        endfor
      endfor
      mus(:, k) = (1/C) .* mus(:, k);
    endfor
    % Sigma
    for k = 1:K
      Sigmas(:, :, k) = zeros(D);
      for n = 1:N
        for l = 1:K^I
          [Z_n, z] = dec2oneofK(l, K, I);
          X_n = X(:, :, n);
          % broadcast the mu_k vector 
          mu_ks = mus(:, k)(:, ones(1, I));
          dev = X_n - mu_ks;
          % the columns of the Kronecker product are the matrices of the outer products of the deviations
          E = reshape(eye(I), 1, I^2);
          Dev = kron(dev, dev);
          % only select the matrices where the inidices of the multiplied vectors are the same (i x i); quasi the diagonal matrices of the Kronecker product
          Dev = Dev(:, logical(E));
          assert(size(Dev, 2) == I);
          % filter the relevant matrices the weights of the latent variable states for component k
          Dev = Dev(:, Z_n(k, :));
          % sum it up and reshape to obtain the unweighted (via posterior) result
          Sigma_data = reshape(sum(Dev, 2), D, D);

          Sigmas(:, :, k) = Sigmas(:, :, k) + p_Z(l, n) .* Sigma_data;
        endfor
      endfor
      % normalize
      C = 0;
      for n = 1:N
        for l = 1:K^I
          [Z_n, z] = dec2oneofK(l, K, I);
          C = C + p_Z(l, n) * sum(Z_n(k, :));
        endfor
      endfor
      Sigmas(:, :, k) = 1/C * Sigmas(:, :, k);
    endfor

    rho
    mus
    Sigmas
    pi

    % compute and monitor for convergence
    delta_mu = 1/(D*K) * norm(mus - mus_prev)
    delta_Sigma = 1/(D*(D-1)*K) * norm(sum(Sigmas - Sigmas_prev, 3))
    delta_pi = 1/K * norm(pi - pi_prev)
    delta_rho = 1/(K^I) * norm(rho - rho_prev)
    % XXX these values are chosen arbitrarily based on the assumption that the input vectors are scaled to be in [0, 1]
    % TODO one might want to use a sliding window to assess the convergence, since EM can have long plateaus
    if(delta_mu < 0.0001 && delta_Sigma < 0.0001 && delta_pi < 0.0001 && delta_rho < 0.0001)
      disp('Breaking due to small changes in model parameters');
      break;
    endif
    disp('M-step toc')
    toc()
  endfor
endfunction





