function [mus, Sigmas, rho, pi] = learnExactIndependent(K, X, d, max_iter)
  disp('Init tic')
  tic()
  [D, I, N] = size(X);
  % starting the learning phase
  % allocate space for the model parameters
  mus = rand(D, K, I);
  Sigmas = eye(D)(:, :, ones(1, K), ones(1, I));
  pi = 1/K*ones(K, I);
  rho = 0.5 * ones(K^I, 1);

  % run k-means to obtain an initial estimate for the mixture components (mean and covariance)
  for i = 1:I
    X_i = reshape(X(:, i, :), D, N);
    [idx, centers] = kmeans(X_i', K);
    mus(:, :, i) = centers';
    % compute the intra-cluster covariance and use it to initialize the component covariance
    for k = 1:K
      Sigmas(:, :, k, i) = cov(X_i(:, idx' == k)');
    endfor
  endfor

  % keep a copy of the previous model params to monitor the progress
  mus_prev = mus;
  Sigmas_prev = Sigmas;
  pi_prev = pi;
  rho_prev = rho;

  disp('Init toc')
  toc()

  % TODO compute one 3D matrix of all states of Z: KxIxK^I
  % this can then be used many times to use vector operations instead of loops

  for it = 1:max_iter
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
    disp('E-step tic');
    tic()
    p_Z = computePosterior(mus, Sigmas, pi, rho, X, d, K);
    toc()
    % tic()
    % p_Z = zeros(K^I, N);
    % for n = 1:N
    %   for l = 1:K^I
    %     [Z_n, z] = dec2oneOfK(l, K, I);
    %     % select the right mus
    %     mus_l = zeros(D, I);
    %     % z_idx = (base2dec(z(1, :)', K)) + 1;
    %     z_idx = zeros(I, 1);
    %     for i = 1:I
    %       z_idx(i) = base2decfast(z(1, i), K) + 1;
    %     endfor
    %     % assert(z_idx == z_idx_fast)
    %     for i = 1:I
    %       mus_l(:, i) = mus(:, z_idx(i), i);
    %     endfor
    %     % select the right Sigmas
    %     Sigmas_l = zeros(D, D, I);
    %     for i = 1:I
    %       Sigmas_l(:, :, i) = Sigmas(:, :, z_idx(i), i);
    %     endfor
    %     % select the right pis
    %     pi_l = pi(logical(Z_n));
    %     % compute the posterior for the current state and observation
    %     p_Z(l, n) = rho(l)^d(n) * (1 - rho(l))^(1 - d(n));
    %     for i = 1:I
    %       p_x_n_i = mvnpdf(X(:, i, n)', mus_l(:, i)', Sigmas_l(:, :, i));
    %       p_Z(l, n) = p_Z(l, n) * pi_l(i) * p_x_n_i;
    %     endfor
    %   endfor
    % endfor
    % % normalize
    % p_Z = p_Z ./ sum(p_Z);
    % sum(sum(abs(p_Z - p_Z_fast)))
    % assert(sum(sum(abs(p_Z - p_Z_fast))) < 0.0001)
    % disp('E-step toc')
    % toc()

    % M-step
    disp('M-step rho')
    tic()
    % rho
    for l = 1:K^I
      rho(l) = p_Z(l, :) * d';
    endfor
    rho = rho ./ sum(p_Z, 2);
    toc()
    % pi, mus
    disp('M-step pi, mus')
    tic()
    [mus, pi] = maxMuPi(p_Z, X, K);
    % toc()
    % tic()
    % pi = pi .* 0;
    % mus = mus .* 0;
    % for i = 1:I
    %   for k = 1:K
    %     % mus
    %     mu_norm = 0;
    %     for n = 1:N
    %       for l = 1:K^I
    %         [Z_n, z] = dec2oneOfK(l, K, I);
    %         % pi
    %         pi(k, i) = pi(k, i) + Z_n(k, i) * p_Z(l, n);
    %         % mus
    %         mus(:, k, i) = mus(:, k, i) + p_Z(l, n) * Z_n(k, i) * X(:, i, n);
    %         mu_norm = mu_norm + p_Z(l, n) * Z_n(k, i);
    %       endfor
    %     endfor
    %     % mus
    %     mus(:, k, i) = mus(:, k, i) ./ mu_norm;
    %   endfor
    % endfor
    % pi = pi ./ N;
    % assert(pi_fast == pi);
    % assert(abs(sum(sum(sum(mus_fast))) - sum(sum(sum(mus)))) < 0.0001);
    toc()
    % XXX There is a data dependency between mus and Sigmas, so they cannot be integrated in the same loop
    % Sigmas
    disp('M-step Sigmas')
    tic()
    [Sigmas_fast] = maxSigmas(X, mus, p_Z);
    toc()
    tic()
    for i = 1:I
      for k = 1:K
        Sigma_norm = 0;
        for n = 1:N
          diff_n = X(:, :, n) - reshape(mus(:, k, :), D, I);
          diff_n
          for l = 1:K^I
            [Z_n, z] = dec2oneOfK(l, K, I);
            Sigmas(:, :, k, i) = Sigmas(:, :, k, i) + p_Z(l, n) * Z_n(k, i) * diff_n(:, i) * diff_n(:, i)';
            Sigma_norm = Sigma_norm + p_Z(l, n) * Z_n(k, i);
          endfor
        endfor
        Sigmas(:, :, k, i) = Sigmas(:, :, k, i) ./ Sigma_norm;
      endfor
    endfor
    toc()
    Sigmas_fast
    Sigmas
    assert(sum(sum(sum(sum(abs(Sigmas - Sigmas_fast))))) < 0.0001)
    
    rho
    mus
    Sigmas
    pi

    % monitor for convergence
    % TODO consider evaluating the changes for each component individually and then require for each component to change a certain amount
    delta_mu = 1/(D * K * I) * norm(reshape(mus - mus_prev, D, K * I))
    %delta_Sigma = 1/(D*(D-1)*K) * norm(sum(Sigmas - Sigmas_prev, 3))
    delta_pi = 1/(K * I) * norm(pi - pi_prev)
    delta_rho = 1/(K^I) * norm(rho - rho_prev)
    % XXX these values are chosen arbitrarily based on the assumption that the input vectors are scaled to be in [0, 1]
    % TODO one might want to use a sliding window to assess the convergence, since EM can have long plateaus
    if(delta_mu < 0.0001 && delta_pi < 0.0001 && delta_rho < 0.0001)
      disp('Breaking due to small changes in model parameters');
      break;
    endif
  endfor
endfunction





