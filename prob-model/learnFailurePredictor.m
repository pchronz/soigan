function [mus, Sigmas, rho, pi] = learnFailurePredictor(K, X, d, max_iter)
  disp('Init tic')
  tic()
  [D, I, N] = size(X);
  % starting the learning phase
  % allocate space for the model parameters
  mus = zeros(D, K);
  Sigmas = eye(D)(:, :, ones(1, K));
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
    M = 200;
    M_over = 10;
    Z_sampled = zeros(K, I, M, N);
    for n = 1:N
      % choose a random initial state
      l = randi(K^I);
      [Z_n_old, z] = dec2oneofK(l, K, I);
      % compute the probability for the current state
      p_Z_old = rho(l)^d(n) * (1-rho(l))^(1-d(n));
      for i = 1:I
        % choose the right mixture component
        mu_k = mus * Z_n_old(:, i);
        Sigma_k = Sigmas(:, :, str2num(z(1, i)) + 1);
        pi_k = pi' * Z_n_old(:, i);
        p_Z_old = p_Z_old * pi_k * mvnpdf(X(:, i, n)', mu_k, Sigma_k);
      endfor
      Z_n_sampled = zeros(K, I, M*M_over);
      for m = 1:(M*M_over)
        % choose another random state sampled from a uniform distribution
        l = randi(K^I);
        [Z_n_new, z] = dec2oneofK(l, K, I);
        % compute the probability for the new state
        p_Z_new = rho(l)^d(n) * (1-rho(l))^(1-d(n));
        for i = 1:I
          % choose the right mixture component
          mu_k = mus * Z_n_new(:, i);
          Sigma_k = Sigmas(:, :, str2num(z(1, i)) + 1);
          pi_k = pi' * Z_n_new(:, i);
          p_Z_new = p_Z_new * pi_k * mvnpdf(X(:, i, n)', mu_k, Sigma_k);
        endfor
        A = 1;
        if(p_Z_old > 0)
          A = min(1, p_Z_new/p_Z_old);
        endif
        % TODO compute hit-rate or the expected acceptance or the average acceptance
        if(rand(1) >= A)
          Z_n_sampled(:, :, m) = Z_n_new;
          Z_n_old = Z_n_new;
          p_Z_old = p_Z_new;
        else
          Z_n_sampled(:, :, m) = Z_n_old;
        endif
      endfor
      % subsample
      sub_idx = logical(repmat([1, zeros(1, M_over - 1)], 1, M));
      Z_sampled(:, :, :, n) = Z_n_sampled(:, :, sub_idx);
    endfor
    disp('E-step toc')
    toc()

    disp('M-step tic')
    tic()
    % M-step
    % rho
    L = zeros(N, M);
    for n = 1:N
      for m = 1:M
        L(n, m) = oneOfK2Dec(Z_sampled(:, :, m, n));
      endfor
    endfor
    l = unique(reshape(L, 1, N*M));
    for el = l
      l_n = sum(double(L == el), 2);
      rho(el) = (l_n' * d') / sum(l_n);
    endfor
    % pi
    pi = 1/(I*N*M) * sum(sum(sum(Z_sampled, 4), 3), 2);
    % mu
    for k = 1:K
      mus(:, k) = zeros(D, 1);
      Z_sampled_summed = reshape(sum(Z_sampled, 3), K, I, N);
      for n = 1:N
        mus(:, k) = mus(:, k) + X(:, :, n) * Z_sampled_summed(k, :, n)';
      endfor
    endfor
    % normalize the mus
    mus = mus * diag(1./(I * N * M * pi'));
    % Sigma
    for k = 1:K
      Sigmas(:, :, k) = zeros(D);
      Z_sampled_summed = reshape(sum(Z_sampled, 3), K, I, N);
      for n = 1:N
        X_n = X(:, :, n);
        Dev_k = X_n - mus(:, k)(:, ones(1, I));
        for i = 1:I
          Sigmas(:, :, k) = Sigmas(:, :, k) + Dev_k(:, i) * Dev_k(:, i)' * Z_sampled_summed(k, i, n);
        endfor
      endfor
      % normalize the Sigmas
      Sigmas(:, :, k) = 1/(I * N * M * pi(k)) * Sigmas(:, :, k);
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





