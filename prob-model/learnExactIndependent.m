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

    empty = true;
    idx = 0;
    centers = 0;
    while(empty)
      try
        [idx, centers] = kmeans(X_i', K);
        empty = false;
      catch
        i
        disp('Got an empty cluster, trying again...');
      end_try_catch
    endwhile
    mus(:, :, i) = centers';
    % compute the intra-cluster covariance and use it to initialize the component covariance
    i
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

    % Cope with singularities.
    % If one of the components has a singular covariance matrix, sample a value from the component with an isotropic covariance in the order of the smalles covariance from all other components.
    % Determine singular components
    for i = 1:I
      for k = 1:K
        Sigma_ki = Sigmas(:, :, k, i);
        % Is Sigma_ki positive definite?
        if(isdefinite(Sigma_ki) != 1)
          disp('Discovered a covariance matrix that is not positive definite.')
          k
          i
          % Compute the parameter for the isotropic Gaussian that minimises D_KL(p||q), based on the Gaussians for the same service i.
          %alphas = inf(1, K);
          %for k2 = 1:K
          %  if (k2 == k)
          %    continue
          %  endif
          %  % TODO Try to compute the parameters by minimising D_KL(q||p) instead, if you can solve the integral. Minimizing D_KL(q||p) should result in a larger variance, while minimizing D_KL(p||q) should keep it quite small. 
          %  alphas(1, k2) = 1/D*trace(Sigmas(:, :, k2, i));
          %end
          %alpha = min(alphas);
          % XXX Do something more meaningful, such as the above.
          % Set the component's covariance to the isotropic covariance.
          Sigmas(:, :, k, i) = eye(D);
          % Sample a value from a multi-variate Gaussian with mu_ki and alpha.
          x_new = mvnrnd(mus(:, k, i)', Sigmas(:, :, k, i));
          % Find out on which observation the component centers on.
          dists = zeros(1, N);
          for n = 1:N
            % What is the distance between the mean and each value?
            dists(1, n) = sqrt((mus(:, k, i) - X(:, i, n))'*(mus(:, k, i) - X(:, i, n)));
          end
          [dist, n_sing] = min(dists);
          % TODO retry removing singular observations instead of sampling more
          % Copy the values of the non-singular components for the same timeframe.
          X_new = X(:, :, n_sing);
          % Replace the observed value for the singular component, with the sampled value.
          X_new(:, i) = x_new;
          % Add the generated observation for all services to the data set.
          X(:, :, end + 1) = X_new;
          % Add the global state data to it.
          d(end + 1) = d(n_sing);
          % Update the data set's size
          [D, I, N] = size(X);
          % Restart the optimization.
          it = 0;
        endif
      end
    end
    

    % E-step
    % compute the posterior for all possible states for Z for all observations N
    % this results in a matrix with (K^I)*N entries, you might want to optimize this at some point
    % nevertheless we will need the posterior many times, so it should help us quite a lot
    % let's get it over with...
    disp('E-step tic');
    tic()
    p_Z = computePosterior(mus, Sigmas, pi, rho, X, d, K);
    if(sum(sum(isnan(p_Z))) != 0)
      p_Z
      Sigmas
      mus
      rho
      pi
    endif
    assert(sum(sum(isnan(p_Z))) == 0)
    toc()
    %tic()
    %p_Z = zeros(K^I, N);
    %for n = 1:N
    %  n
    %  for l = 1:K^I
    %    [Z_n, z] = dec2oneOfK(l, K, I);
    %    % select the right mus
    %    mus_l = zeros(D, I);
    %    % z_idx = (base2dec(z(1, :)', K)) + 1;
    %    z_idx = zeros(I, 1);
    %    for i = 1:I
    %      z_idx(i) = base2decfast(z(1, i), K) + 1;
    %    endfor
    %    % assert(z_idx == z_idx_fast)
    %    for i = 1:I
    %      mus_l(:, i) = mus(:, z_idx(i), i);
    %    endfor
    %    % select the right Sigmas
    %    Sigmas_l = zeros(D, D, I);
    %    for i = 1:I
    %      Sigmas_l(:, :, i) = Sigmas(:, :, z_idx(i), i);
    %    endfor
    %    % select the right pis
    %    pi_l = pi(logical(Z_n));
    %    % compute the posterior for the current state and observation
    %    p_Z(l, n) = rho(l)^d(n) * (1 - rho(l))^(1 - d(n));
    %    for i = 1:I
    %      p_x_n_i = mvnpdf(X(:, i, n)', mus_l(:, i)', Sigmas_l(:, :, i));
    %      p_Z(l, n) = p_Z(l, n) * pi_l(i) * p_x_n_i;
    %    endfor
    %  endfor
    %endfor
    %% normalize
    %p_Z = p_Z ./ sum(p_Z);
    %sum(sum(abs(p_Z - p_Z_fast)))
    %assert(sum(sum(abs(p_Z - p_Z_fast))) < 0.0001)
    %disp('E-step toc')
    %toc()

    % M-step
    disp('M-step rho')
    tic()
    % rho
    for l = 1:K^I
      rho(l) = p_Z(l, :) * d';
    endfor
    sum_p_Z = sum(p_Z, 2);
    
    % Test whether the probabilities over the latent variables approximately sum to 1.
    assert(sum(sum(p_Z) >= 1.0001) == 0 && sum(sum(p_Z) < 0.9999) == 0)
    rho = rho ./ sum_p_Z;
    % Test whether all rho values are ok.
    assert(sum(isnan(rho)) == 0)
    toc()
    % pi, mus
    disp('M-step pi, mus')
    tic()
    [mus, pi] = maxMuPi(p_Z, X, K);
    % Test whether the probabilities over the mixture components sum approximately to 1. 
    assert(sum(sum(pi) >= 1.0001) == 0 && sum(sum(pi) <= 0.9999) == 0)
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
    [Sigmas] = maxSigmas(X, mus, p_Z);
    toc()
    %tic()
    %Sigmas = Sigmas .* 0;
    %for i = 1:I
    %  for k = 1:K
    %    Sigma_norm = 0;
    %    for n = 1:N
    %      diff_n = X(:, :, n) - reshape(mus(:, k, :), D, I);
    %      for l = 1:K^I
    %        [Z_n, z] = dec2oneOfK(l, K, I);
    %        Sigmas(:, :, k, i) = Sigmas(:, :, k, i) + p_Z(l, n) * Z_n(k, i) * diff_n(:, i) * diff_n(:, i)';
    %        Sigma_norm = Sigma_norm + p_Z(l, n) * Z_n(k, i);
    %      endfor
    %    endfor
    %    Sigmas(:, :, k, i) = Sigmas(:, :, k, i) ./ Sigma_norm;
    %  endfor
    %endfor
    %toc()
    %assert(sum(sum(sum(sum(abs(Sigmas - Sigmas_fast))))) < 0.0001)

    rho
    mus
    Sigmas
    pi

    % monitor for convergence
    % TODO consider evaluating the changes for each component individually and then require for each component to change a certain amount
    % TODO try out Frobenius norm instead. I am currently not too sure what the p=2 norm really means. p=2 norm is the Euclidean norm.
    delta_mu = 1/(D * K * I) * norm(reshape(mus - mus_prev, D, K * I))
    %delta_Sigma = 1/(D*D*K*I) * norm(sum(Sigmas - Sigmas_prev))
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





